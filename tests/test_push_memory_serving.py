"""Tiny fixed-input nearest serving through real fake-native Push IO.

One cached nearest initializer and one two-row append; no neural initialization,
fit, optimizer, teacher, SDK or physics. Predictions use genuine nearest math.
"""

from collections import defaultdict
from dataclasses import replace
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts import push_cube_continuation_study as study
from src.adapters import numpy_continuous_bc as bp
from src.adapters import numpy_nearest_continuous_bc as nearest
from src.core import continuous_bc as core
from src.core.cube_taskspace_state import (
    CubeTaskSpaceState,
    cube_taskspace_observation,
    cube_taskspace_spec,
)
from src.core.pick_cube_action_history import ActionHistoryState
from src.infra import maniskill_push_cube as native
from test_push_cube_continuous import FakePushEnvironment


def forbidden(*args, **options):
    pytest.fail("Push memory fixture admitted neural work, fitting or SDK")


def assert_same(first, second):
    assert first.policy == second.policy
    assert (first.completed_updates, first.training_examples) == (
        second.completed_updates,
        second.training_examples,
    )
    np.testing.assert_array_equal(first.prototype_features, second.prototype_features)
    np.testing.assert_array_equal(first.prototype_actions, second.prototype_actions)


@pytest.fixture(scope="module", autouse=True)
def ledger():
    counts: defaultdict[str, int] = defaultdict(int)
    initialize, append, predict = (
        nearest.NumpyNearestContinuousBCLearner.__init__,
        nearest.NumpyNearestContinuousBCLearner.train_batch,
        nearest.NumpyNearestContinuousBCLearner.predict_with_work,
    )

    def initialized(owner, policy):
        counts["nearest_initializations"] += 1
        return initialize(owner, policy)

    def appended(owner, features, targets):
        counts["append_attempts"] += 1
        counts["append_rows"] += len(features)
        result = append(owner, features, targets)
        counts["append_returns"] += 1
        return result

    def predicted(owner, features):
        counts["query_attempts"] += 1
        counts["query_rows"] += len(features)
        result = predict(owner, features)
        counts["query_returns"] += 1
        counts["distance_candidates"] += result.distance_candidates
        counts["distance_feature_values"] += result.distance_feature_values
        return result

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(nearest.NumpyNearestContinuousBCLearner, "__init__", initialized)
        patch.setattr(nearest.NumpyNearestContinuousBCLearner, "train_batch", appended)
        patch.setattr(nearest.NumpyNearestContinuousBCLearner, "predict_with_work", predicted)
        patch.setattr(bp.NumpyContinuousBCLearner, "__init__", forbidden)
        patch.setattr(bp.NumpyContinuousBCLearner, "train_batch", forbidden)
        patch.setattr(core, "_forward", forbidden)
        patch.setattr(core, "continuous_bc_loss_and_gradients", forbidden)
        patch.setattr(bp, "fit_training_standardization", forbidden)
        patch.setattr(native, "_make_environment", forbidden)
        yield counts
    if os.environ.get("PUSH_MEMORY_SERVING_LEDGER"):
        with Path(os.environ["PUSH_MEMORY_SERVING_LEDGER"]).open("x", encoding="utf8") as output:
            json.dump(dict(counts), output, sort_keys=True)
            output.write("\n")
    assert (
        counts["nearest_initializations"]
        == counts["append_attempts"]
        == counts["append_returns"]
        == 1
    )
    assert counts["append_rows"] == 2
    assert counts["query_attempts"] == counts["query_returns"] == counts["query_rows"] == 11
    assert counts["distance_candidates"] == 22 and counts["distance_feature_values"] == 264
    assert counts["factory_attempts"] == 12 and counts["factory_returns"] == 11
    assert counts["control_attempts"] == 7 and counts["control_returns"] == 6
    assert counts["grasp_attempts"] == counts["grasp_returns"] == 17
    assert counts["close_attempts"] == counts["close_returns"] == 11


@pytest.fixture(scope="module")
def owner(ledger):
    owner = nearest.NumpyNearestContinuousBCLearner(
        nearest.FixedInputNearestContinuousBCPolicy(cube_taskspace_spec())
    )
    command = (float(np.float32(0.123456789)), 0.0, 0.0, -1.0)
    states = (
        CubeTaskSpaceState(False, (0.0, 0.0, 0.1), (0.05, 0.0, 0.02), (0.3, 0.0, 0.02)),
        CubeTaskSpaceState(True, (0.01, 0.0, 0.1), (0.05, 0.0, 0.02), (0.3, 0.0, 0.02)),
    )
    features = np.asarray(
        [
            cube_taskspace_observation(states[0]),
            cube_taskspace_observation(states[1], ActionHistoryState(command, 1)),
        ],
        np.float64,
    )
    owner.train_batch(
        features, np.asarray([(0.123456789, 0.0, 0.0, -1.0), (-0.25, 0.0, 0.0, 1.0)], np.float64)
    )
    return owner


@pytest.fixture
def source(owner):
    return owner.fork()


@pytest.fixture
def boundary(monkeypatch, ledger):
    environment = FakePushEnvironment(
        ledger, horizon=2, terminated=False, truncated=True, success=False
    )
    mode: dict[str, Any] = {}

    def factory():
        ledger["factory_attempts"] += 1
        if mode.get("factory"):
            raise RuntimeError("fake factory unavailable")
        if mode.get("on_factory"):
            mode["on_factory"]()
        ledger["factory_returns"] += 1
        return environment

    monkeypatch.setattr(native, "_make_environment", factory)
    return environment, mode


def test_should_serve_actual_two_row_choices_float32_history_and_complete_negative(
    source, boundary
):
    environment, _ = boundary
    before = source.snapshot_state()
    report = study.roll_push(source, label="tiny-memory")
    assert report["complete"] is True, report
    assert (
        report["final_success"] is report["any_success"] is False and report["sparse_return"] == 0
    )
    assert report["source_updates"] == 1 and report["source_training_rows"] == 2
    assert report["source_unchanged"] is True
    assert report["nearest_selected_indices"] == [0, 1]
    assert report["nearest_prediction_attempts"] == report["nearest_prediction_returns"] == 2
    assert (
        report["nearest_distance_candidate_upper_bound"]
        == report["nearest_distance_candidates_completed"]
        == 4
    )
    assert (
        report["nearest_distance_feature_upper_bound"]
        == report["nearest_distance_feature_values_completed"]
        == 48
    )
    assert (
        report["unreturned_prediction_possible"]
        is report["unvalidated_nearest_prediction_work_unknown"]
        is False
    )
    assert len(report["nearest_prediction_reports"]) == 2
    rows = report["episode"]["transitions"]
    command = float(np.float32(0.123456789))
    assert tuple(rows[0]["action"]) == (command, 0.0, 0.0, -1.0)
    assert tuple(rows[1]["observation"])[7:] == (command, 0.0, 0.0, -1.0, 1 / 6)
    assert rows[-1]["truncated"] is True and rows[-1]["terminated"] is False
    assert environment.reset_seeds == [17] and environment.close_calls == 1
    assert report["work"]["completed_steps"] == 2
    assert report["work"]["completed_control_physics_steps"] == 10
    assert_same(source.snapshot_state(), before)


@pytest.mark.parametrize("fault", ["array_owner", "policy_owner"])
def test_should_refuse_equal_value_native_owner_replacement_before_query(source, boundary, fault):
    environment, mode = boundary

    def changed():
        if fault == "array_owner":
            object.__setattr__(
                source._state, "prototype_features", source._state.prototype_features.copy()
            )
        else:
            source._policy = replace(source._policy)

    mode["on_factory"] = changed
    report = study.roll_push(source, label="changed-owner")
    assert (
        report["complete"] is False
        and report["prediction_attempts"] == report["control_attempts"] == 0
    )
    assert environment.close_calls == 1 and report["source_unchanged"] is False


@pytest.mark.parametrize("fault", ["source", "work", "index"])
def test_should_retain_paid_native_receipt_before_refusing_source_or_work(
    source, boundary, monkeypatch, fault
):
    environment, _ = boundary
    predict = nearest.NumpyNearestContinuousBCLearner.predict_with_work

    def changed(owner, features):
        result = predict(owner, features)
        if fault == "source":
            source._state.prototype_features[0, 0] += 0.125
        elif fault == "work":
            result = replace(result, distance_candidates=3)
        else:
            result = replace(result, selected_indices=(2,))
        return result

    monkeypatch.setattr(nearest.NumpyNearestContinuousBCLearner, "predict_with_work", changed)
    report = study.roll_push(source, label="paid-refused")
    assert report["complete"] is False and report["prediction_returns"] == 1
    assert len(report["nearest_prediction_reports"]) == len(report["prediction_outputs"]) == 1
    assert report["control_attempts"] == 0 and environment.close_calls == 1
    assert report["nearest_distance_candidates_completed"] == 0
    assert report["unvalidated_nearest_prediction_work_unknown"] is True


def test_should_keep_unreturned_prediction_unknown_without_fabricating_receipt(
    source, boundary, monkeypatch
):
    predict = nearest.NumpyNearestContinuousBCLearner.predict_with_work

    def unavailable(owner, features):
        predict(owner, features)
        raise RuntimeError("paid retrieval return unavailable")

    monkeypatch.setattr(nearest.NumpyNearestContinuousBCLearner, "predict_with_work", unavailable)
    report = study.roll_push(source, label="lost-return")
    assert report["complete"] is False and report["prediction_attempts"] == 1
    assert report["prediction_returns"] == report["control_attempts"] == 0
    assert (
        report["nearest_prediction_reports"] == []
        and report["unreturned_prediction_possible"] is True
    )


@pytest.mark.parametrize("fault", ["actions", "counter"])
def test_should_refuse_changed_preceding_retrieval_after_one_paid_control(
    source, boundary, monkeypatch, fault
):
    environment, _ = boundary
    returned: list[Any] = []
    predict = nearest.NumpyNearestContinuousBCLearner.predict_with_work

    def captured(owner, features):
        result = predict(owner, features)
        returned.append(result)
        return result

    def changed():
        if fault == "actions":
            returned[0].actions[0, 0] += 0.125
        else:
            object.__setattr__(returned[0], "distance_candidates", 99)

    environment.after_step = changed
    monkeypatch.setattr(nearest.NumpyNearestContinuousBCLearner, "predict_with_work", captured)
    before = source.snapshot_state()
    report = study.roll_push(source, label="prior-receipt")
    assert report["complete"] is False
    assert report["prediction_attempts"] == report["prediction_returns"] == 1
    assert report["control_returns"] == 1 and len(report["transitions"]) == 1
    assert len(report["nearest_prediction_reports"]) == 1 and environment.close_calls == 1
    assert_same(source.snapshot_state(), before)


def test_should_preserve_native_attempt_and_nearest_charge_when_control_raises(source, boundary):
    environment, _ = boundary
    environment.step_error = RuntimeError("fake native control unavailable")
    report = study.roll_push(source, label="failed-control")
    assert (
        report["complete"] is False
        and report["control_attempts"] == 1
        and report["control_returns"] == 0
    )
    assert (
        report["prediction_returns"] == 1 and report["nearest_distance_candidates_completed"] == 2
    )
    assert report["work"]["step_attempts"] == 1 and report["work"]["completed_steps"] == 0
    assert report["unreturned_native_work_unknown"] is True
    assert len(report["push_cube_attempted_actions"]) == 1
    assert report["push_cube_returned_actions"] == () and environment.close_calls == 1


def test_should_preserve_completed_negative_controls_when_cleanup_fails(
    source, boundary, monkeypatch
):
    environment, _ = boundary
    close = environment.close

    def failed():
        close()
        raise RuntimeError("fake cleanup return unavailable")

    monkeypatch.setattr(environment, "close", failed)
    report = study.roll_push(source, label="failed-close")
    assert report["complete"] is False and "close_error" in report
    assert report["control_returns"] == report["prediction_returns"] == 2
    assert report["final_success"] is False and report["nearest_distance_candidates_completed"] == 4


def test_should_keep_factory_failure_unknown_without_prediction_or_control(source, boundary):
    _, mode = boundary
    mode["factory"] = True
    report = study.roll_push(source, label="failed-factory")
    assert report["complete"] is False and report["unreturned_native_work_unknown"] is True
    assert report["prediction_attempts"] == report["control_attempts"] == 0


@pytest.mark.parametrize("fault", ["role", "spec"])
def test_should_refuse_nontraining_or_wrong_spec_before_factory(source, monkeypatch, fault):
    if fault == "role":
        object.__setattr__(source._policy, "training_role", "probe_test")
    else:
        object.__setattr__(
            source._policy,
            "spec",
            replace(
                source._policy.spec, action_units=("other",) + source._policy.spec.action_units[1:]
            ),
        )
    monkeypatch.setattr(native, "_make_environment", forbidden)
    with pytest.raises(ValueError):
        study.roll_push(source, label="refused")
