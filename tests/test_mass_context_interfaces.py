"""Owned mass metadata through teacher/rollout fake IO, without learning."""

from dataclasses import FrozenInstanceError, asdict, replace
import json
import sys

import numpy as np
import pytest

from scripts import maniskill_continuous_smoke as smoke
from scripts import maniskill_task_teacher_smoke as teacher_cli
from src.adapters.numpy_continuous_bc import ContinuousBCPolicy, NumpyContinuousBCLearner
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)
from src.core.pick_cube_camera_context import pick_cube_camera_context
from src.core.pick_cube_mass_context import pick_cube_mass_context
from src.core.pick_cube_teacher import PickCubeGeometry
from src.infra import maniskill_continuous as port
from src.infra import maniskill_task_teacher as teacher
from test_maniskill_bc_smoke import assert_same_state
from test_pick_cube_mass_context import CONTEXT, MassEnvironment


def state_spec():
    return ContinuousActionSpec(42, port.ACTION_UNITS, (-1.0,) * 4, (1.0,) * 4)


def qualified_result(*, mass=True):
    context = CONTEXT if mass else "pick-cube"
    identity = f"{context}-teacher" if mass else "pick-cube-task-teacher-17"
    observations = tuple(tuple(float(index + step) for index in range(42)) for step in range(3))
    episode = ContinuousEpisode(
        identity,
        context,
        "train",
        state_spec(),
        tuple(
            ContinuousTransition(
                i, observations[i], (0.0,) * 4, float(i), observations[i + 1], i == 1, False
            )
            for i in range(2)
        ),
    )
    options = dict(
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
    )
    if mass:
        context_record = pick_cube_mass_context()
        options.update(
            mass_context=context_record,
            mass_readback=port.ManiSkillMassReadback(context_record, 0.125, 0.25),
            mass_work=port.ManiSkillMassWork(8, 8, 1, 1),
        )
    return teacher.TaskTeacherResult(**options)


class InterfaceEnvironment(MassEnvironment):
    """Keep public state42 and sparse success; no geometry/physics surrogate."""

    def __init__(self):
        super().__init__()
        self.reset_observation = np.arange(42, dtype=np.float32)[None]
        self.close_callback = None

    def step(self, command):
        _, _, terminated, truncated, info = super().step(command)
        index = len(self.commands)
        return (
            self.reset_observation + index,
            np.array([float(index == self.steps)], dtype=np.float32),
            terminated,
            truncated,
            info,
        )

    def close(self):
        super().close()
        if self.close_callback is not None:
            self.close_callback()


@pytest.fixture
def boundary(monkeypatch, tmp_path):
    environment, calls = InterfaceEnvironment(), []

    def factory(**kwargs):
        calls.append(kwargs)
        return environment

    monkeypatch.setattr(port, "_make_environment", factory)
    monkeypatch.setattr(smoke, "version", lambda _: "fake-mass-interface")
    monkeypatch.setattr(teacher_cli, "version", lambda _: "fake-mass-interface")
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
    # Tagged counters/momentum are restored metadata, never trained in this file.
    model = NumpyContinuousBCLearner(ContinuousBCPolicy(state_spec(), hidden_dim=2, seed=17))
    initial = model.snapshot_state()
    parameters = tuple(np.zeros_like(value) for value in initial.parameters)
    parameters[3][:] = (0.2, -0.3, 0.4, -0.5)
    model.restore_state(
        replace(
            initial,
            parameters=parameters,
            velocity=tuple(np.full_like(value, 0.125) for value in initial.velocity),
            completed_updates=13,
            training_examples=377,
        )
    )
    return model


def test_should_validate_and_own_exact_original_readback_and_bounded_work():
    context = pick_cube_mass_context()
    original = port.ManiSkillMassReadback(context, 0.125, 0.25)
    owned = port.validate_mass_readback(original, context)
    assert owned == original and owned is not original and owned.context is not context
    object.__setattr__(original.context, "seed", 19)
    assert owned.context == pick_cube_mass_context()
    with pytest.raises(FrozenInstanceError):
        owned.source_mass_kg = 0.25
    work = port.ManiSkillMassWork(9, 8, 1, 1)
    copied = port.validate_mass_work(work)
    assert copied == work and copied is not work
    assert port.validate_mass_work(port.ManiSkillMassWork(0, 0, 0, 0)) == port.ManiSkillMassWork(
        0, 0, 0, 0
    )


@pytest.mark.parametrize(
    "changes",
    [
        dict(source_mass_kg=True),
        dict(source_mass_kg=0),
        dict(source_mass_kg=np.float64(0.125)),
        dict(source_mass_kg=0.0),
        dict(source_mass_kg=-0.125),
        dict(source_mass_kg=0.75),
        dict(source_mass_kg=float("nan")),
        dict(applied_mass_kg=float("inf")),
        dict(applied_mass_kg=0.3),
        dict(applied_mass_kg=0.25000000000000006),
        dict(context=replace(pick_cube_mass_context(), seed=17.0)),
        dict(source_mass_kg=0.1, applied_mass_kg=0.2),
    ],
)
def test_should_refuse_wrong_type_nonfinite_unrepresentable_or_nondoubled_readback(changes):
    value = replace(port.ManiSkillMassReadback(pick_cube_mass_context(), 0.125, 0.25), **changes)
    with pytest.raises(ValueError):
        port.validate_mass_readback(value, pick_cube_mass_context())


@pytest.mark.parametrize(
    "changes",
    [
        dict(read_attempts=True),
        dict(read_returns=1.0),
        dict(read_attempts=-1),
        dict(read_returns=4),
        dict(write_attempts=2),
        dict(write_returns=2),
        dict(write_attempts=0, write_returns=1),
        dict(write_returns=-1),
        dict(read_attempts=2**63),
    ],
)
def test_should_refuse_impossible_or_untyped_mass_work(changes):
    with pytest.raises(ValueError):
        port.validate_mass_work(replace(port.ManiSkillMassWork(3, 3, 1, 1), **changes))


@pytest.mark.parametrize(
    "kind", ["readback_mapping", "work_mapping", "readback_subclass", "work_subclass"]
)
def test_should_refuse_nonexact_mass_metadata_records(kind):
    value = port.ManiSkillMassReadback(pick_cube_mass_context(), 0.125, 0.25)
    work = port.ManiSkillMassWork(3, 3, 1, 1)
    if kind.endswith("subclass"):

        class Readback(port.ManiSkillMassReadback):
            pass

        class Work(port.ManiSkillMassWork):
            pass

        value, work = Readback(value.context, 0.125, 0.25), Work(3, 3, 1, 1)
    else:
        value, work = asdict(value), asdict(work)
    with pytest.raises(ValueError):
        if kind.startswith("readback"):
            port.validate_mass_readback(value, pick_cube_mass_context())
        else:
            port.validate_mass_work(work)


def test_should_collect_qualified_owned_mass_teacher_with_actual_port(boundary):
    environment, calls, _ = boundary
    original = pick_cube_mass_context()
    result = teacher.collect_task_teacher(mass_context=original)
    assert result.qualified and result.error is None
    assert result.observed_success is result.final_success is True
    assert environment.reset_calls == [17] and environment.close_calls == 1
    assert calls == [{}] and result.creation_attempts == 1
    assert result.geometry_attempts == result.geometry_returns == result.teacher_actions == 2
    assert result.stages == ("align", "align")
    assert result.mass_context == original and result.mass_context is not original
    assert (
        result.mass_readback.source_mass_kg == 0.125
        and result.mass_readback.applied_mass_kg == 0.25
    )
    assert result.mass_work.write_attempts == result.mass_work.write_returns == 1
    assert result.mass_work.read_attempts == result.mass_work.read_returns
    episode = result.training_episode()
    assert episode.episode_id == f"{CONTEXT}-teacher" and episode.context == CONTEXT
    assert episode.role == "train" and episode.spec == state_spec()
    assert episode is not result.candidate_episode
    object.__setattr__(original, "seed", 19)
    assert result.training_episode() == episode
    object.__setattr__(episode, "context", "relabelled")
    assert result.training_episode().context == CONTEXT


def test_should_admit_valid_synthetic_teacher_and_own_training_payload():
    result = qualified_result()
    assert result.qualified is True
    extracted = result.training_episode()
    assert extracted == result.candidate_episode and extracted is not result.candidate_episode
    object.__setattr__(extracted.transitions[0], "reward", 0.5)
    assert result.training_episode().transitions[0].reward == 0.0


@pytest.mark.parametrize(
    "damage",
    [
        "missing_context",
        "missing_readback",
        "missing_work",
        "wrong_readback",
        "wrong_identity",
        "wrong_context",
        "heldout_role",
        "dimension",
        "action_units",
        "nonsparse",
        "final_reward",
        "guard_seed",
        "guard_arrival",
        "observed_type",
        "final_type",
        "camera",
        "missing_export_read",
        "failed_read",
        "extra_write",
        "controller_count",
        "all_metadata_removed",
        "creation_type",
        "geometry_type",
        "stage_type",
    ],
)
def test_should_refuse_lost_relabelled_or_corrupted_mass_teacher_qualification(damage):
    result = qualified_result()
    episode = result.candidate_episode
    if damage in ("missing_context", "missing_readback", "missing_work"):
        result = replace(
            result,
            **{
                dict(
                    missing_context="mass_context",
                    missing_readback="mass_readback",
                    missing_work="mass_work",
                )[damage]: None
            },
        )
    elif damage == "wrong_readback":
        result = replace(result, mass_readback=replace(result.mass_readback, applied_mass_kg=0.5))
    elif damage in ("missing_export_read", "failed_read", "extra_write"):
        result = replace(
            result,
            mass_work=replace(
                result.mass_work,
                **dict(
                    missing_export_read=dict(read_attempts=7, read_returns=7),
                    failed_read=dict(read_attempts=9),
                    extra_write=dict(write_attempts=2, write_returns=2),
                )[damage],
            ),
        )
    elif damage == "all_metadata_removed":
        result = replace(result, mass_context=None, mass_readback=None, mass_work=None)
    elif damage in ("creation_type", "geometry_type", "stage_type"):
        result = replace(
            result,
            **dict(
                creation_type=dict(creation_attempts=True),
                geometry_type=dict(geometry_returns=2.0),
                stage_type=dict(stages=["align", "align"]),
            )[damage],
        )
    elif damage == "controller_count":
        result = replace(result, work=replace(result.work, completed_steps=1))
    elif damage in ("wrong_identity", "wrong_context", "heldout_role"):
        field, value = dict(
            wrong_identity=("episode_id", "pick-cube-task-teacher-17"),
            wrong_context=("context", "pick-cube"),
            heldout_role=("role", "probe_train"),
        )[damage]
        episode = replace(episode, **{field: value})
        result = replace(result, candidate_episode=episode)
    elif damage in ("dimension", "action_units"):
        spec = episode.spec
        if damage == "dimension":
            rows = tuple(
                replace(
                    row, observation=row.observation[:3], next_observation=row.next_observation[:3]
                )
                for row in episode.transitions
            )
            spec = replace(spec, observation_dim=3)
        else:
            rows = episode.transitions
            spec = replace(spec, action_units=("other",) + spec.action_units[1:])
        result = replace(result, candidate_episode=replace(episode, spec=spec, transitions=rows))
    elif damage in ("nonsparse", "final_reward"):
        index, reward = (0, 0.5) if damage == "nonsparse" else (1, 0.0)
        rows = list(episode.transitions)
        rows[index] = replace(rows[index], reward=reward)
        result = replace(result, candidate_episode=replace(episode, transitions=tuple(rows)))
    else:
        result = replace(
            result,
            **dict(
                guard_seed=dict(guard_seed=31),
                guard_arrival=dict(labels_arrived_at=2),
                observed_type=dict(observed_success=1),
                final_type=dict(final_success=1),
                camera=dict(camera_context=pick_cube_camera_context()),
            )[damage],
        )
    assert result.qualified is False
    with pytest.raises(ValueError):
        result.training_episode()


@pytest.mark.parametrize("failure", ["geometry", "native_step", "late_mass", "cleanup"])
def test_should_keep_teacher_paid_work_and_unqualified_partial_failure(
    boundary, monkeypatch, failure
):
    environment, _, _ = boundary
    if failure == "geometry":
        monkeypatch.setattr(
            teacher, "_geometry", lambda _: (_ for _ in ()).throw(RuntimeError("geometry failed"))
        )
    elif failure == "native_step":
        environment.step_error = RuntimeError("native failed")
    elif failure == "late_mass":
        environment.after_step = lambda: environment.cube.value.fill(0.5)
    else:
        original = port.ManiSkillPickCubeEpisode.mass_readback.fget
        monkeypatch.setattr(
            port.ManiSkillPickCubeEpisode,
            "mass_readback",
            property(
                lambda owner: (
                    replace(original(owner), applied_mass_kg=0.5)
                    if environment.close_calls
                    else original(owner)
                )
            ),
        )
    result = teacher.collect_task_teacher(mass_context=pick_cube_mass_context())
    assert not result.qualified and result.error is not None
    assert environment.close_calls == 1 and result.creation_attempts == 1
    expected = dict(geometry=0, native_step=1, late_mass=1, cleanup=2)[failure]
    assert len(environment.commands) == result.work.step_attempts == expected
    assert result.work.completed_steps == (
        0 if failure in ("geometry", "native_step") else expected
    )
    assert result.mass_readback is not None and result.mass_work is not None
    with pytest.raises(ValueError):
        result.training_episode()


@pytest.mark.parametrize(
    "changes",
    [
        dict(train_seed=19),
        dict(train_seed=True),
        dict(camera_context=pick_cube_camera_context(), visual_observation=True),
        dict(mass_context={}),
    ],
)
def test_should_reject_invalid_teacher_mass_request_before_factory(boundary, changes):
    _, calls, _ = boundary
    options = dict(mass_context=pick_cube_mass_context())
    options.update(changes)
    with pytest.raises(ValueError):
        teacher.collect_task_teacher(**options)
    assert calls == []


@pytest.mark.parametrize("damage", ["readback", "work", "context"])
def test_should_reject_teacher_callback_changes_before_controls_and_preserve_original(
    boundary, monkeypatch, damage
):
    environment, _, _ = boundary

    def geometry(owner):
        if damage == "readback":
            owner._mass_source_mass, owner._mass_applied_mass = 0.25, 0.5
        elif damage == "work":
            owner._mass_reads = owner._mass_read_returns = 2
        else:
            owner._context = "pick-cube"
        return PickCubeGeometry((0.0,) * 3, (0.02, -0.03, 0.01), (0.05, 0.02, 0.08))

    monkeypatch.setattr(teacher, "_geometry", geometry)
    result = teacher.collect_task_teacher(mass_context=pick_cube_mass_context())
    assert result.qualified is False and result.error is not None
    assert result.geometry_attempts == result.geometry_returns == 1
    assert result.teacher_actions == 0 and environment.commands == []
    assert result.mass_readback.source_mass_kg == 0.125
    assert result.mass_readback.applied_mass_kg == 0.25
    assert result.work.step_attempts == result.work.completed_steps == 0


def test_should_run_actual_frozen_mass_predictions_and_keep_all_source_arrays(boundary, source):
    environment, calls, directory = boundary
    before = source.snapshot_state()
    result = smoke.run_frozen_rollout(
        directory / "mass.json", learner=source, mass_context=pick_cube_mass_context()
    )
    report = result.report
    assert report["complete"] is True and report["mass_context_exact"] is True
    assert report["context"] == CONTEXT and report["seed"] == 17
    assert report["mass_context"] == asdict(pick_cube_mass_context())
    assert report["mass_readback"] == dict(
        context=asdict(pick_cube_mass_context()), source_mass_kg=0.125, applied_mass_kg=0.25
    )
    assert report["mass_work"]["write_returns"] == 1
    assert report["prediction_attempts"] == report["prediction_returns"] == 2
    assert report["validated_prediction_rows"] == 2 and report["validated_prediction_bytes"] == 64
    assert report["source_updates"] == 13 and report["source_training_rows"] == 377
    assert report["optimizer_updates"] == 0 and result.corrections is None
    expected = np.tanh(np.array([[0.2, -0.3, 0.4, -0.5]])).astype(np.float32)
    for command in environment.commands:
        np.testing.assert_array_equal(command, expected)
    assert calls == [{}] and environment.close_calls == 1
    assert report["original_owner_exact"] is report["frozen_state_exact"] is True
    assert_same_state(before, source.snapshot_state())
    assert json.loads((directory / "mass.json").read_text()) == json.loads(json.dumps(report))


def test_should_preserve_unreturned_native_work_and_prediction_receipt(boundary, source):
    environment, _, directory = boundary
    before = source.snapshot_state()
    environment.step_error = RuntimeError("native failed after admission")
    report = smoke.run_frozen_rollout(
        directory / "failed-step.json", learner=source, mass_context=pick_cube_mass_context()
    ).report
    assert report["complete"] is False
    assert (
        report["prediction_attempts"]
        == report["prediction_returns"]
        == report["validated_prediction_rows"]
        == 1
    )
    assert report["validated_prediction_bytes"] == 32
    assert report["work"] == asdict(port.ManiSkillWork(1, 0, 0, 5))
    assert report["mass_readback"]["source_mass_kg"] == 0.125
    assert report["mass_readback"]["applied_mass_kg"] == 0.25
    assert report["mass_work"]["read_attempts"] == 4
    assert len(environment.commands) == environment.close_calls == 1
    assert_same_state(before, source.snapshot_state())


@pytest.mark.parametrize("phase", ["after_control", "cleanup"])
@pytest.mark.parametrize("damage", ["readback", "context", "work"])
def test_should_refuse_late_mass_metadata_and_clear_success_without_losing_work(
    boundary, source, monkeypatch, phase, damage
):
    environment, _, directory = boundary
    before = source.snapshot_state()
    property_name = dict(readback="mass_readback", context="mass_context", work="mass_work")[damage]
    original = getattr(port.ManiSkillPickCubeEpisode, property_name).fget

    def getter(owner):
        value = original(owner)
        corrupt = (
            bool(environment.commands)
            if phase == "after_control"
            else bool(environment.close_calls)
        )
        if corrupt:
            if damage == "readback":
                return replace(value, source_mass_kg=0.25, applied_mass_kg=0.5)
            if damage == "context":
                return replace(value, seed=17.0)
            return replace(value, write_returns=True)
        return value

    monkeypatch.setattr(port.ManiSkillPickCubeEpisode, property_name, property(getter))
    report = smoke.run_frozen_rollout(
        directory / "corrupt.json", learner=source, mass_context=pick_cube_mass_context()
    ).report
    assert report["complete"] is False and report["mass_context_exact"] is not True
    paid = 1 if phase == "after_control" else 2
    assert len(environment.commands) == report["work"]["completed_steps"] == paid
    assert report["prediction_attempts"] == report["prediction_returns"] == paid
    assert report["validated_prediction_rows"] == paid
    assert environment.close_calls == 1
    assert_same_state(before, source.snapshot_state())


@pytest.mark.parametrize(
    "changes",
    [
        dict(train_seed=19),
        dict(mass_context={}),
        dict(camera_context=pick_cube_camera_context(), visual_observation=True),
        dict(corrective_queries=True),
    ],
)
def test_should_refuse_invalid_rollout_mass_before_fork_or_factory(
    boundary, source, monkeypatch, changes
):
    _, calls, directory = boundary
    monkeypatch.setattr(
        NumpyContinuousBCLearner, "fork", lambda _: pytest.fail("invalid mass forked learner")
    )
    options = dict(learner=source, mass_context=pick_cube_mass_context())
    options.update(changes)
    report = smoke.run_frozen_rollout(directory / "invalid.json", **options).report
    assert report["complete"] is False and report["creation_attempts"] == 0
    assert report["prediction_attempts"] == 0 and calls == []


def test_should_recheck_source_after_mass_getter_callback_before_control(
    boundary, source, monkeypatch
):
    environment, _, directory = boundary
    before = source.snapshot_state()
    original_predict = NumpyContinuousBCLearner.predict
    original_readback = port.ManiSkillPickCubeEpisode.mass_readback.fget
    returned = []

    def predict(model, features):
        values = original_predict(model, features)
        returned.append(True)
        return values

    def readback(owner):
        value = original_readback(owner)
        if returned:
            source._state.parameters[0][0, 0] = 99.0
        return value

    monkeypatch.setattr(NumpyContinuousBCLearner, "predict", predict)
    monkeypatch.setattr(port.ManiSkillPickCubeEpisode, "mass_readback", property(readback))
    try:
        report = smoke.run_frozen_rollout(
            directory / "owner-corrupt.json", learner=source, mass_context=pick_cube_mass_context()
        ).report
        assert report["complete"] is False and report["original_owner_exact"] is not True
        assert report["prediction_attempts"] == report["prediction_returns"] == 1
        assert report["validated_prediction_rows"] == 1
        assert report["work"]["step_attempts"] == 0 and environment.commands == []
    finally:
        source.restore_state(before)
    assert_same_state(before, source.snapshot_state())


@pytest.mark.parametrize("phase", ["before_control", "final"])
def test_should_recheck_owned_mass_after_frozen_state_callback_before_control(
    boundary, source, monkeypatch, phase
):
    environment, _, directory = boundary
    before = source.snapshot_state()
    original_init = port.ManiSkillPickCubeEpisode.__init__
    original_predict = NumpyContinuousBCLearner.predict
    original_snapshot = NumpyContinuousBCLearner.snapshot_state
    original_readback = port.ManiSkillPickCubeEpisode.mass_readback.fget
    ports, predicted, armed, mutated = [], [], [], []

    def initialize(owner, *args, **kwargs):
        original_init(owner, *args, **kwargs)
        ports.append(owner)

    def predict(owner, features):
        values = original_predict(owner, features)
        predicted.append(True)
        return values

    def readback(owner):
        value = original_readback(owner)
        if predicted:
            armed.append(True)
        return value

    def snapshot(owner):
        value = original_snapshot(owner)
        admitted_phase = phase == "before_control" or environment.close_calls > 0
        if owner is source and armed and not mutated and admitted_phase:
            ports[0]._mass_context = None
            mutated.append(True)
        return value

    monkeypatch.setattr(port.ManiSkillPickCubeEpisode, "__init__", initialize)
    monkeypatch.setattr(port.ManiSkillPickCubeEpisode, "mass_readback", property(readback))
    monkeypatch.setattr(NumpyContinuousBCLearner, "predict", predict)
    monkeypatch.setattr(NumpyContinuousBCLearner, "snapshot_state", snapshot)
    report = smoke.run_frozen_rollout(
        directory / "binding-corrupt.json", learner=source, mass_context=pick_cube_mass_context()
    ).report
    assert mutated == [True] and report["complete"] is False
    assert report["mass_context_exact"] is not True
    predictions, controls = (1, 0) if phase == "before_control" else (2, 2)
    assert report["prediction_attempts"] == report["prediction_returns"] == predictions
    assert report["validated_prediction_rows"] == predictions
    assert report["work"]["step_attempts"] == report["work"]["completed_steps"] == controls
    assert len(environment.commands) == controls
    assert report["original_owner_exact"] is report["frozen_state_exact"] is True
    assert_same_state(before, source.snapshot_state())


def test_should_preserve_default_json_fieldsets_and_exclusive_output(boundary):
    _, calls, directory = boundary
    output = directory / "default.json"
    report = smoke.run_frozen_rollout(output).report
    assert report["complete"] is True
    assert not {"mass_context", "mass_readback", "mass_work", "mass_context_exact"} & report.keys()
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        smoke.run_frozen_rollout(output, mass_context=pick_cube_mass_context())
    assert output.read_bytes() == original and len(calls) == 1


@pytest.mark.parametrize("entry", ["teacher", "rollout"])
def test_should_forward_explicit_cli_mass_and_keep_default_teacher_schema(
    boundary, source, monkeypatch, entry
):
    _, _, directory = boundary
    output = directory / "cli.json"
    monkeypatch.setattr(sys, "argv", [entry, "--output", str(output), "--mass-drift"])
    if entry == "rollout":
        assert smoke.main(learner=source) == 0
    else:
        assert teacher_cli.main() == 0
    report = json.loads(output.read_text())
    assert report["mass_context"] == asdict(pick_cube_mass_context())
    assert report["context"] == CONTEXT and report["seed"] == 17
    if entry == "teacher":
        assert report["qualified"] is True and report["optimizer_updates"] == 0


def test_should_keep_default_teacher_cli_new_fields_absent(boundary, monkeypatch):
    _, _, directory = boundary
    output = directory / "teacher-default.json"
    monkeypatch.setattr(teacher_cli, "collect_task_teacher", lambda: qualified_result(mass=False))
    monkeypatch.setattr(sys, "argv", ["teacher", "--output", str(output)])
    assert teacher_cli.main() == 0
    report = json.loads(output.read_text())
    assert not {"mass_context", "mass_readback", "mass_work"} & report.keys()


@pytest.mark.parametrize(
    "flags",
    [
        ["--train-seed", "19"],
        ["--guard-seed", "31", "--labels-arrived-at", "2"],
        ["--camera-drift", "--visual-observation"],
    ],
)
def test_should_refuse_invalid_teacher_cli_allocation_before_output(boundary, monkeypatch, flags):
    _, calls, directory = boundary
    output = directory / "invalid-cli.json"
    monkeypatch.setattr(sys, "argv", ["teacher", "--output", str(output), "--mass-drift", *flags])
    with pytest.raises(SystemExit) as raised:
        teacher_cli.main()
    assert raised.value.code == 2 and not output.exists() and calls == []
