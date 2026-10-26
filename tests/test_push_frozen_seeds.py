"""Fresh TRAIN reset binding with fixed serving owners and bounded fake IO.

32 cases: eight valid routes,22 prework seed refusals and two paid nondefault
failures. One nearest constructor/append2rows,5 genuine queries10 candidates120
feature visits. BP predictions are counted doubles, with zero neural arithmetic.
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
from src.core.cube_taskspace_state import cube_taskspace_spec
from src.infra import maniskill_push_cube as native
from test_push_cube_continuous import FakePushEnvironment


def forbidden(*args, **options):
    pytest.fail("invalid seed reached snapshot, clock, model or native work")


@pytest.fixture(scope="module", autouse=True)
def ledger():
    counts: defaultdict[str, int] = defaultdict(int)
    initialize = nearest.NumpyNearestContinuousBCLearner.__init__
    append = nearest.NumpyNearestContinuousBCLearner.train_batch
    predict = nearest.NumpyNearestContinuousBCLearner.predict_with_work

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
        patch.setattr(bp, "fit_training_standardization", forbidden)
        patch.setattr(core, "_forward", forbidden)
        patch.setattr(core, "continuous_bc_loss_and_gradients", forbidden)
        patch.setattr(native, "_make_environment", forbidden)
        yield counts
    if os.environ.get("PUSH_FROZEN_SEEDS_LEDGER"):
        with Path(os.environ["PUSH_FROZEN_SEEDS_LEDGER"]).open("x", encoding="utf8") as output:
            json.dump(dict(counts), output, sort_keys=True)
            output.write("\n")
    assert (
        counts["nearest_initializations"]
        == counts["append_attempts"]
        == counts["append_returns"]
        == 1
    )
    assert counts["append_rows"] == 2
    assert counts["query_attempts"] == counts["query_returns"] == counts["query_rows"] == 5
    assert counts["distance_candidates"] == 10 and counts["distance_feature_values"] == 120
    assert counts["bp_prediction_attempts"] == counts["bp_prediction_returns"] == 5
    for kind in ("factory", "reset", "close"):
        assert counts[kind + "_attempts"] == counts[kind + "_returns"] == 10
    assert counts["control_attempts"] == counts["control_returns"] == 8
    assert counts["grasp_attempts"] == counts["grasp_returns"] == 18


@pytest.fixture(scope="module")
def memory(ledger):
    owner = nearest.NumpyNearestContinuousBCLearner(
        nearest.FixedInputNearestContinuousBCPolicy(cube_taskspace_spec())
    )
    # Same feature rows deliberately exercise stable earliest-prototype ties.
    owner.train_batch(
        np.zeros((2, 12), np.float64),
        np.asarray([(0.123456789, 0.0, 0.0, -1.0), (-0.25, 0.0, 0.0, 1.0)], np.float64),
    )
    return owner


def controlled_bp():
    owner = object.__new__(bp.NumpyContinuousBCLearner)
    owner._policy = bp.ContinuousBCPolicy(cube_taskspace_spec(), hidden_dim=2, seed=17)
    parameters = (
        np.zeros((12, 2), np.float64),
        np.zeros(2, np.float64),
        np.zeros((2, 4), np.float64),
        np.zeros(4, np.float64),
    )
    owner._state = bp.ContinuousBCSnapshot(
        owner._policy,
        parameters,
        (parameters[0].copy(), parameters[1].copy(), parameters[2].copy(), parameters[3].copy()),
        40,
        2776,
    )
    return owner


@pytest.fixture
def boundary(monkeypatch, ledger):
    environment = FakePushEnvironment(
        ledger, horizon=1, terminated=False, truncated=True, success=False
    )
    mode: dict[str, Any] = {}

    def factory():
        ledger["factory_attempts"] += 1
        ledger["factory_returns"] += 1
        return environment

    def predict(owner, features):
        ledger["bp_prediction_attempts"] += 1
        assert features.dtype == np.float64 and features.shape == (1, 12)
        result = np.asarray([(0.123456789, 0.0, 0.0, -1.0)], np.float64)
        ledger["bp_prediction_returns"] += 1
        if mode.get("source_change"):
            owner._state.parameters[3][0] += 0.125
        return result

    monkeypatch.setattr(native, "_make_environment", factory)
    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "predict", predict)
    return environment, mode


@pytest.mark.parametrize("family", ["bp", "nearest"])
@pytest.mark.parametrize(
    "selected", [None, 17, 19, 20], ids=["default17", "explicit17", "train19", "train20"]
)
def test_should_bind_only_reset_seed_while_preserving_frozen_source_and_native_history(
    memory, boundary, family, selected
):
    environment, _ = boundary
    owner = memory.fork() if family == "nearest" else controlled_bp()
    before = owner.snapshot_state()
    options = {} if selected is None else dict(seed=selected)
    expected = 17 if selected is None else selected
    result = study.roll_push(owner, label="fixed-owner", **options)
    assert result["complete"] is True, result
    assert result["seed"] == result["metadata"]["seed"] == expected
    assert environment.reset_seeds == [expected] and environment.close_calls == 1
    assert (
        result["metadata"]["context"]
        == result["episode"]["context"]
        == f"push-cube-train-{expected}"
    )
    assert result["episode"]["episode_id"] == "push-frozen-fixed-owner"
    assert result["role"] == result["episode"]["role"] == "train"
    assert result["source_unchanged"] is True
    assert (
        result["final_success"] is result["any_success"] is False and result["sparse_return"] == 0
    )
    assert (
        result["prediction_attempts"]
        == result["prediction_returns"]
        == result["control_returns"]
        == 1
    )
    row = result["transitions"][0]
    command = float(np.float32(0.123456789))
    assert tuple(row["action"]) == (command, 0.0, 0.0, -1.0)
    assert tuple(row["next_observation"])[7:] == (command, 0.0, 0.0, -1.0, 1 / 6)
    assert row["truncated"] is True and row["terminated"] is False
    after = owner.snapshot_state()
    assert before.policy == after.policy
    if family == "nearest":
        assert result["nearest_selected_indices"] == [0]
        assert result["nearest_distance_candidates_completed"] == 2
        assert result["nearest_distance_feature_values_completed"] == 24
        np.testing.assert_array_equal(after.prototype_features, before.prototype_features)
        np.testing.assert_array_equal(after.prototype_actions, before.prototype_actions)
        assert (after.completed_updates, after.training_examples) == (1, 2)
    else:
        assert "nearest_prediction_reports" not in result
        assert study.same_continuous_state(before, after)


@pytest.mark.parametrize("family", ["bp", "nearest"])
@pytest.mark.parametrize("seed", [True, False, 17.0, None, "19", 18, 21, 23, 24, 31, 37])
def test_should_refuse_unsupported_seed_before_any_source_or_time_or_port_access(
    monkeypatch, family, seed
):
    owner = object.__new__(
        bp.NumpyContinuousBCLearner if family == "bp" else nearest.NumpyNearestContinuousBCLearner
    )
    monkeypatch.setattr(study, "_pin", forbidden)
    monkeypatch.setattr(study, "_pin_push_nearest", forbidden)
    monkeypatch.setattr(study, "monotonic", forbidden)
    monkeypatch.setattr(study, "ManiSkillPushCubeEpisode", forbidden)
    with pytest.raises(ValueError, match="seed"):
        study.roll_push(owner, label="refused", seed=seed)


def test_should_keep_nondefault20_paid_nearest_receipt_and_close_before_refusing_work(
    memory, boundary, monkeypatch
):
    environment, _ = boundary
    owner = memory.fork()
    predict = nearest.NumpyNearestContinuousBCLearner.predict_with_work

    def changed(source, features):
        return replace(predict(source, features), distance_candidates=3)

    monkeypatch.setattr(nearest.NumpyNearestContinuousBCLearner, "predict_with_work", changed)
    result = study.roll_push(owner, label="bad-receipt", seed=20)
    assert result["complete"] is False and result["seed"] == 20
    assert result["prediction_returns"] == 1 and result["control_attempts"] == 0
    assert len(result["nearest_prediction_reports"]) == len(result["prediction_outputs"]) == 1
    assert result["unvalidated_nearest_prediction_work_unknown"] is True
    assert result["nearest_distance_candidates_completed"] == 0
    assert environment.reset_seeds == [20] and environment.close_calls == 1
    assert result["work"]["close_returns"] == 1


def test_should_preserve_nondefault19_paid_bp_output_and_real_source_mutation_without_control(
    boundary,
):
    environment, mode = boundary
    owner = controlled_bp()
    mode["source_change"] = True
    result = study.roll_push(owner, label="changed-source", seed=19)
    assert result["complete"] is False and result["source_unchanged"] is False
    assert result["prediction_returns"] == 1 and len(result["prediction_outputs"]) == 1
    assert result["control_attempts"] == 0 and owner._state.parameters[3][0] == 0.125
    assert environment.reset_seeds == [19] and environment.close_calls == 1
    assert result["work"]["close_returns"] == 1
