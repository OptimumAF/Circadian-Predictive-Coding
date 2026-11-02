"""Literal maps/fake IO only: zero solves, updates, model draws or physics."""

from dataclasses import FrozenInstanceError, replace
from collections import defaultdict
from typing import Any

import numpy as np
import pytest

from scripts import maniskill_continuous_smoke as smoke
from src.adapters.readonly_continuous_affine import ReadonlyContinuousAffine
from src.core import continuous_affine_fit as core
from src.core import pick_cube_action_history as history
from src.core.cube_taskspace_state import cube_taskspace_spec
from src.core.pick_cube_taskspace_history import PickCubeTaskSpaceHistoryRecipe
from src.infra import maniskill_continuous as native
from test_maniskill_continuous import FakeEnvironment

WORK = dict(matmuls=0, prediction_rows=0)


@pytest.fixture(autouse=True)
def no_fit_and_bounded_predictions(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("readonly tests must not solve, fit or create random models")

    monkeypatch.setattr(np.linalg, "lstsq", forbidden)
    monkeypatch.setattr(core, "_solve", forbidden)
    original = core.np.matmul
    rows = 0

    def bounded(left, right):
        nonlocal rows
        rows += len(left)
        assert rows <= 32
        WORK["matmuls"] += 1
        WORK["prediction_rows"] += len(left)
        assert WORK["prediction_rows"] <= 512
        return original(left, right)

    monkeypatch.setattr(core.np, "matmul", bounded)


def model():
    coefficients = [[0.0] * 4 for _ in range(13)]
    coefficients[12] = [0.125, -0.3, 2.0, -1.0]
    coefficients[7][0] = 0.5
    return core.ContinuousAffineFit(
        cube_taskspace_spec(),
        tuple(tuple(row) for row in coefficients),
        198,
        13,
        (1.0,) * 13,
        1.0,
        1.0,
    )


def test_should_predict_clipped_physical_actions_without_update_authority():
    source = model()
    reader = ReadonlyContinuousAffine(source)
    before = reader.snapshot_state()
    x = np.zeros((2, 12), dtype=np.float64)
    x[1, 7] = 0.25
    actions = reader.predict(x)
    np.testing.assert_array_equal(actions, [[0.125, -0.3, 1.0, -1.0], [0.25, -0.3, 1.0, -1.0]])
    actions.fill(0)
    assert reader.snapshot_state() == before
    assert reader.policy.definition == core.DEFINITION
    assert before.completed_updates == 0 and before.training_examples == 198
    assert not hasattr(reader, "train") and not hasattr(reader, "restore_state")
    with pytest.raises(FrozenInstanceError):
        setattr(reader, "_state", before)
    # External source/snapshot records do not own inference state.
    object.__setattr__(source, "fit_rows", 1)
    object.__setattr__(before.model, "fit_rows", 2)
    assert reader.snapshot_state().model.fit_rows == 198


@pytest.mark.parametrize("fault", ["state", "coefficients", "role", "rank"])
def test_should_refuse_replacement_or_corruption_without_repair(fault):
    reader = ReadonlyContinuousAffine(model())
    if fault == "state":
        object.__setattr__(reader, "_state", reader.snapshot_state())
    elif fault == "coefficients":
        values = tuple(tuple(list(row)) for row in reader._state.model.coefficients)
        object.__setattr__(reader._state.model, "coefficients", values)
    elif fault == "role":
        object.__setattr__(reader._state.policy, "training_role", "final")
    else:
        object.__setattr__(reader._state.model, "rank", 12)
    with pytest.raises(ValueError):
        reader.predict(np.zeros((1, 12), dtype=np.float64))


def test_should_make_a_separate_readonly_fork_and_detect_private_changes():
    reader = ReadonlyContinuousAffine(model())
    original = smoke._prepare_learner(reader)
    forked = reader.fork()
    assert forked is not reader and forked._state is not reader._state
    result: dict[str, Any] = {}
    smoke._check_state_projection_frozen(reader, forked, original, result)
    assert result == dict(frozen_state_exact=True, original_owner_exact=True)
    object.__setattr__(forked._state.model, "fit_rows", 197)
    with pytest.raises(ValueError):
        smoke._check_frozen(reader, forked, original, result)
    assert result["frozen_state_exact"] is None
    assert reader.snapshot_state() == original


def raw(index):
    values = [0.0] * 42
    values[7:9] = [0.02, 0.02]
    values[18] = float(index > 0)
    values[22:26] = [-0.5, 0.5, -0.5, 0.5]
    values[32:36] = [0.0, 0.0, -1.0, 0.0]
    values[36:42] = [-0.02 + index * 0.005, 0.01, -0.1, 0.03, -0.04, 0.15]
    return tuple(values)


class TaskspaceEnvironment(FakeEnvironment):
    def __init__(self, after_step=None):
        super().__init__()
        self.reset_observation = np.asarray([raw(0)], dtype=np.float64)
        self.after_step = after_step

    def step(self, command):
        result = super().step(command)
        if self.after_step is not None:
            self.after_step()
        return (np.asarray([raw(len(self.commands))], dtype=np.float64), *result[1:])


@pytest.mark.parametrize("seed", [17, 19, 20])
def test_should_drive_existing_rollout_using_actual_executed_history(monkeypatch, tmp_path, seed):
    reader = ReadonlyContinuousAffine(model())
    initial = reader.snapshot_state()
    environment = TaskspaceEnvironment()
    monkeypatch.setattr(native, "_make_environment", lambda: environment)
    monkeypatch.setattr(smoke, "version", lambda _: "literal-affine-boundary")
    recipe = PickCubeTaskSpaceHistoryRecipe()
    seen = []
    original_predict = ReadonlyContinuousAffine.predict

    def predict(owner, x):
        seen.append(x.copy())
        return original_predict(owner, x)

    monkeypatch.setattr(ReadonlyContinuousAffine, "predict", predict)
    outcome = smoke.run_frozen_rollout(
        tmp_path / "roll.json", learner=reader, train_seed=seed, action_history_recipe=recipe
    )
    report = outcome.report
    assert report["complete"] is True, report
    assert report["policy"] == "frozen_readonly_continuous_affine"
    assert report["source_affine_fit_rows"] == 198
    assert report["optimizer_updates"] == report["source_updates"] == 0
    assert report["prediction_attempts"] == report["prediction_returns"] == 2
    assert report["frozen_state_exact"] is report["original_owner_exact"] is True
    assert report["action_history_exact"] is True
    assert environment.close_calls == 1 and environment.reset_calls == [seed]
    assert len(environment.commands) == len(seen) == 2
    prior = history.advance_action_history(
        history.ActionHistoryState(), tuple(float(v) for v in environment.commands[0][0])
    )
    assert tuple(seen[0][0]) == history.action_history_observation(
        raw(0), history.ActionHistoryState(), recipe
    )
    assert tuple(seen[1][0]) == history.action_history_observation(raw(1), prior, recipe)
    assert seen[1][0, 7] == float(environment.commands[0][0, 0])
    assert reader.snapshot_state() == initial


@pytest.mark.parametrize("option", ["missing", "zero", "raw47", "queries", "probe", "decoding"])
def test_should_refuse_unsupported_authority_before_native_creation(monkeypatch, tmp_path, option):
    creations = []
    monkeypatch.setattr(native, "_make_environment", lambda: creations.append(True))
    recipe = PickCubeTaskSpaceHistoryRecipe()
    kwargs: dict[str, Any] = dict(
        learner=ReadonlyContinuousAffine(model()), action_history_recipe=recipe
    )
    if option == "missing":
        kwargs["action_history_recipe"] = None
    elif option == "zero":
        kwargs["action_history_recipe"] = replace(recipe, mode="zero")
    elif option == "raw47":
        kwargs["action_history_recipe"] = history.PickCubeActionHistoryRecipe()
    elif option == "queries":
        kwargs["corrective_queries"] = True
    elif option == "probe":
        kwargs["probe_test_seed"] = 24
    else:
        kwargs["action_decoding_recipe"] = object()
    result = smoke.run_frozen_rollout(tmp_path / "rejected.json", **kwargs)
    assert result.report["complete"] is False
    assert creations == []


def test_should_stop_after_source_replacement_before_another_command(monkeypatch, tmp_path):
    reader = ReadonlyContinuousAffine(model())

    def corrupt():
        object.__setattr__(reader, "_state", reader.snapshot_state())

    environment = TaskspaceEnvironment(corrupt)
    monkeypatch.setattr(native, "_make_environment", lambda: environment)
    monkeypatch.setattr(smoke, "version", lambda _: "literal-affine-boundary")
    result = smoke.run_frozen_rollout(
        tmp_path / "failed.json",
        learner=reader,
        action_history_recipe=PickCubeTaskSpaceHistoryRecipe(),
    )
    assert result.report["complete"] is False
    assert len(environment.commands) == 1 and environment.close_calls == 1


def test_should_drive_existing_push_port_without_refitting(monkeypatch):
    from scripts import push_cube_continuation_study as study
    from src.infra import maniskill_push_cube as push
    from test_push_cube_continuous import FakePushEnvironment

    reader = ReadonlyContinuousAffine(model())
    before = reader.snapshot_state()
    environment = FakePushEnvironment(defaultdict(int))
    monkeypatch.setattr(push, "_make_environment", lambda: environment)
    report = study.roll_push(reader, label="literal-affine")
    assert report["complete"] is True, report
    assert report["source_unchanged"] is True
    assert report["policy"] == "frozen_readonly_continuous_affine"
    assert report["optimizer_updates"] == 0
    assert report["prediction_attempts"] == report["prediction_returns"] == 2
    assert report["control_attempts"] == report["control_returns"] == 2
    assert reader.snapshot_state() == before
    assert environment.close_calls == 1


@pytest.mark.parametrize("owner", ["source", "private"])
@pytest.mark.parametrize("phase", ["before_command", "final"])
def test_should_refuse_owner_replaced_by_last_history_observer(monkeypatch, tmp_path, owner, phase):
    reader = ReadonlyContinuousAffine(model())
    environment = TaskspaceEnvironment()
    monkeypatch.setattr(native, "_make_environment", lambda: environment)
    monkeypatch.setattr(smoke, "version", lambda _: "literal-affine-boundary")
    private = []
    original_fork = ReadonlyContinuousAffine.fork

    def fork(source):
        result = original_fork(source)
        private.append(result)
        return result

    monkeypatch.setattr(ReadonlyContinuousAffine, "fork", fork)
    check = smoke._check_action_history_port
    fired: list[bool] = []
    predicted_observers = []
    closed_observers = []

    def late_observer(port, encoding, report, **kwargs):
        check(port, encoding, report, **kwargs)
        if report["prediction_returns"] == 1 and len(environment.commands) == 0:
            predicted_observers.append(True)
        if environment.close_calls == 1:
            closed_observers.append(True)
        ready = (
            len(predicted_observers) == 2
            if phase == "before_command"
            else len(closed_observers) == 2
        )
        if ready and not fired:
            target = reader if owner == "source" else private[0]
            object.__setattr__(target, "_state", target.snapshot_state())
            fired.append(True)

    monkeypatch.setattr(smoke, "_check_action_history_port", late_observer)
    outcome = smoke.run_frozen_rollout(
        tmp_path / "late.json",
        learner=reader,
        action_history_recipe=PickCubeTaskSpaceHistoryRecipe(),
    )
    assert fired == [True]
    assert outcome.report["complete"] is False
    assert outcome.report["action_history_exact"] is None
    assert not (
        outcome.report["frozen_state_exact"] is True
        and outcome.report["original_owner_exact"] is True
    )
    assert len(environment.commands) == (0 if phase == "before_command" else 2)
    assert environment.close_calls == 1
