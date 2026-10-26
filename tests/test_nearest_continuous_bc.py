"""Independent bounded nearest-action checks; fake IO is not robot skill evidence."""

from dataclasses import replace
import json
import sys

import numpy as np
import pytest

from test_maniskill_bc_smoke import FakeEnvironment
from scripts import maniskill_continuous_smoke as smoke
from src.adapters import numpy_nearest_continuous_bc as adapter
from src.core import nearest_continuous_bc as core
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)
from src.core.observation_standardization import ObservationStandardization
from src.infra import maniskill_continuous as port


def spec():
    return ContinuousActionSpec(2, ("m/s", "N"), (-2.0, 4.0), (6.0, 8.0))


def rows():
    return (
        np.array([[10.0, 0.0], [0.0, 2.0]], dtype=np.float64),
        np.array([[-2.0, 4.0], [6.0, 8.0]], dtype=np.float64),
    )


def policy(capacity=256, role="train", transform=None):
    frozen = (
        ObservationStandardization((0.0, 0.0), (10.0, 1.0), 2) if transform is None else transform
    )
    return adapter.NearestContinuousBCPolicy(spec(), frozen, capacity, role)


def learner(capacity=256, role="train"):
    return adapter.NumpyNearestContinuousBCLearner(policy(capacity, role))


def episode(role="train", declared=None):
    return ContinuousEpisode(
        "synthetic",
        "nearest",
        role,
        spec() if declared is None else declared,
        (
            ContinuousTransition(0, (10.0, 0.0), (-2.0, 4.0), 0.0, (0.0, 2.0), False, False),
            ContinuousTransition(1, (0.0, 2.0), (6.0, 8.0), 1.0, (1.0, 2.0), False, True),
        ),
    )


def assert_same_state(first, second):
    assert first.policy == second.policy
    assert first.completed_updates == second.completed_updates
    assert first.training_examples == second.training_examples
    np.testing.assert_array_equal(first.prototype_features, second.prototype_features)
    np.testing.assert_array_equal(first.prototype_actions, second.prototype_actions)


def test_should_match_hand_nearest_indices_actions_and_earliest_ties():
    prototypes = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 0.0]])
    actions = np.array([[-2.0, 4.0], [6.0, 8.0], [1.0, 5.0]])
    query = np.array([[0.0, 0.0], [2.0, 0.0], [3.0, 0.0], [1.0, 0.0]])
    result = core.nearest_continuous_actions(query, prototypes, actions, spec())
    assert result.selected_indices == (0, 1, 1, 0)
    np.testing.assert_array_equal(
        result.actions, [[-2.0, 4.0], [6.0, 8.0], [6.0, 8.0], [-2.0, 4.0]]
    )
    assert result.distance_candidates == 12 and result.distance_feature_values == 24
    result.actions.fill(999.0)
    np.testing.assert_array_equal(actions, [[-2.0, 4.0], [6.0, 8.0], [1.0, 5.0]])


def test_should_use_squared_l2_when_l1_would_choose_a_different_row():
    result = core.nearest_continuous_actions(
        np.array([[0.0, 0.0]]),
        np.array([[3.0, 0.0], [2.0, 2.0]]),
        np.array([[-2.0, 4.0], [6.0, 8.0]]),
        spec(),
    )
    assert result.selected_indices == (1,)
    np.testing.assert_array_equal(result.actions, [[6.0, 8.0]])


def test_should_use_frozen_normalized_coordinates_instead_of_raw_distance_and_report_pair_additions():
    features, actions = rows()
    query = np.array([[0.0, 0.0]])
    assert core.nearest_continuous_actions(query, features, actions, spec()).selected_indices == (
        1,
    )
    model = learner()
    update = model.train_batch(features, actions)
    assert update.definition == "continuous_nearest_bc_added_pairs_v1" and update.value == 2
    result = model.predict_with_work(query)
    assert result.selected_indices == (0,)
    np.testing.assert_array_equal(result.actions, [[-2.0, 4.0]])
    np.testing.assert_array_equal(
        model.snapshot_state().prototype_features, [[1.0, 0.0], [0.0, 2.0]]
    )
    assert result.distance_candidates == 2 and result.distance_feature_values == 4
    before = model.snapshot_state()
    np.testing.assert_array_equal(model.predict(query), result.actions)
    assert_same_state(before, model.snapshot_state())


def test_should_append_without_eviction_and_keep_first_action_for_duplicate_observation():
    model = learner(3)
    first_features, first_actions = rows()
    model.train_batch(first_features, first_actions)
    model.train_batch(first_features[:1], np.array([[3.0, 5.0]]))
    result = model.predict_with_work(first_features[:1])
    assert result.selected_indices == (0,)
    np.testing.assert_array_equal(result.actions, first_actions[:1])
    state = model.snapshot_state()
    assert state.completed_updates == 2 and state.training_examples == 3
    np.testing.assert_array_equal(state.prototype_actions, [[-2.0, 4.0], [6.0, 8.0], [3.0, 5.0]])


def test_should_own_training_data_snapshots_policy_and_fork_continuation():
    transform = ObservationStandardization((0.0, 0.0), (10.0, 1.0), 2)
    model = adapter.NumpyNearestContinuousBCLearner(policy(transform=transform))
    features, actions = rows()
    model.train_batch(features, actions)
    before = model.snapshot_state()
    features.fill(999.0)
    actions.fill(999.0)
    object.__setattr__(transform, "scales", (999.0, 1.0))
    assert_same_state(before, model.snapshot_state())
    forked, restored = model.fork(), learner()
    restored.restore_state(before)
    external_policy = model.policy
    object.__setattr__(external_policy.observation_standardization, "means", (999.0, 0.0))
    before.prototype_features.fill(999.0)
    before.prototype_actions.fill(999.0)
    for item in (model, forked, restored):
        item.train_batch(np.array([[5.0, 1.0]]), np.array([[2.0, 6.0]]))
    assert_same_state(model.snapshot_state(), forked.snapshot_state())
    assert_same_state(model.snapshot_state(), restored.snapshot_state())
    untouched = model.snapshot_state()
    forked.train_batch(np.array([[7.0, 1.0]]), np.array([[3.0, 7.0]]))
    assert_same_state(untouched, model.snapshot_state())


def test_should_refuse_cold_prediction_and_capacity_overflow_without_publication():
    model = learner(2)
    empty = model.snapshot_state()
    assert empty.prototype_features.shape == (0, 2) and empty.prototype_actions.shape == (0, 2)
    assert empty.completed_updates == empty.training_examples == 0
    with pytest.raises(ValueError):
        model.predict(np.array([[0.0, 0.0]]))
    assert_same_state(empty, model.snapshot_state())
    features, actions = rows()
    model.train_batch(features, actions)
    before = model.snapshot_state()
    with pytest.raises(ValueError):
        model.train_batch(features[:1], actions[:1])
    assert_same_state(before, model.snapshot_state())


def test_should_refuse_over_capacity_before_normalizing_and_refuse_failed_normalization_atomically(
    monkeypatch,
):
    model = learner(2)
    model.train_batch(*rows())
    before = model.snapshot_state()
    with monkeypatch.context() as patch:
        patch.setattr(
            adapter.observations,
            "standardize_observations",
            lambda *_: pytest.fail("over-capacity append reached preprocessing"),
        )
        with pytest.raises(ValueError):
            features, actions = rows()
            model.train_batch(features[:1], actions[:1])
    assert_same_state(before, model.snapshot_state())
    transform = ObservationStandardization((-1e308, 0.0), (1.0, 1.0), 1)
    unrepresentable = adapter.NumpyNearestContinuousBCLearner(policy(transform=transform))
    empty = unrepresentable.snapshot_state()
    with pytest.raises((ValueError, FloatingPointError)):
        unrepresentable.train_batch(np.array([[1e308, 0.0]]), np.array([[0.0, 5.0]]))
    assert_same_state(empty, unrepresentable.snapshot_state())


@pytest.mark.parametrize("role", ["probe_train", "probe_test", "final_test"])
def test_should_refuse_wrong_episode_roles_without_changing_state(role):
    model = learner()
    before = model.snapshot_state()
    with pytest.raises(ValueError):
        model.train_episode(episode(role))
    assert_same_state(before, model.snapshot_state())


def test_should_accept_only_matching_probe_training_policy_and_matching_spec():
    model = learner(role="probe_train")
    assert model.train_episode(episode("probe_train")).value == 2
    before = model.snapshot_state()
    changed = replace(spec(), action_units=("m", "N"))
    with pytest.raises(ValueError):
        model.train_episode(episode("probe_train", changed))
    assert_same_state(before, model.snapshot_state())


@pytest.mark.parametrize("capacity", [0, 257, True, 2.0, np.int64(2)])
def test_should_refuse_invalid_capacity(capacity):
    with pytest.raises(ValueError):
        policy(capacity)


@pytest.mark.parametrize("bad", [None, "foreign", "dimension", "corrupt"])
def test_should_require_exact_valid_dimension_matching_frozen_statistics(bad):
    transform = ObservationStandardization((0.0, 0.0), (1.0, 1.0), 2)
    if bad is None:
        transform = None
    elif bad == "foreign":
        transform = object()
    elif bad == "dimension":
        transform = ObservationStandardization((0.0,), (1.0,), 2)
    else:
        object.__setattr__(transform, "scales", (0.0, 1.0))
    with pytest.raises(ValueError):
        adapter.NearestContinuousBCPolicy(spec(), transform)


@pytest.mark.parametrize("argument", ["query", "features", "actions"])
@pytest.mark.parametrize("bad", ["list", "float32", "nan", "flat", "shape", "empty", "many"])
def test_should_refuse_ambiguous_or_unbounded_core_arrays(argument, bad):
    features, actions = rows()
    values = {"query": np.array([[0.0, 0.0]]), "features": features, "actions": actions}
    item = values[argument]
    if bad == "list":
        item = item.tolist()
    elif bad == "float32":
        item = item.astype(np.float32)
    elif bad == "nan":
        item[0, 0] = np.nan
    elif bad == "flat":
        item = item.ravel()
    elif bad == "shape":
        item = np.zeros((len(item), 1), dtype=np.float64)
    elif bad == "empty":
        item = item[:0]
    else:
        item = np.zeros((257, 2), dtype=np.float64)
    values[argument] = item
    with pytest.raises(ValueError):
        core.nearest_continuous_actions(
            values["query"], values["features"], values["actions"], spec()
        )


def test_should_refuse_out_of_bounds_actions_before_distance_math():
    features, actions = rows()
    actions[0, 0] = -2.00000001
    with pytest.raises(ValueError):
        core.nearest_continuous_actions(features[:1], features, actions, spec())


@pytest.mark.parametrize(
    "query,prototype",
    [([1e308, 0.0], [-1e308, 0.0]), ([1e200, 0.0], [0.0, 0.0]), ([1e154, 1e154], [0.0, 0.0])],
)
def test_should_refuse_unrepresentable_difference_square_or_distance_sum(query, prototype):
    with pytest.raises((ValueError, FloatingPointError)):
        core.nearest_continuous_actions(
            np.array([query]), np.array([prototype]), np.array([[0.0, 5.0]]), spec()
        )


def test_should_accept_maximum_feature_action_prototype_and_query_bounds():
    declared = ContinuousActionSpec(1024, ("unit",) * 16, (-1.0,) * 16, (1.0,) * 16)
    result = core.nearest_continuous_actions(
        np.zeros((1, 1024)), np.zeros((256, 1024)), np.zeros((256, 16)), declared
    )
    assert result.selected_indices == (0,) and result.distance_feature_values == 256 * 1024
    declared = ContinuousActionSpec(1, ("unit",), (-1.0,), (1.0,))
    result = core.nearest_continuous_actions(
        np.zeros((256, 1)), np.zeros((1, 1)), np.zeros((1, 1)), declared
    )
    assert result.selected_indices == (0,) * 256 and result.distance_candidates == 256


@pytest.mark.parametrize(
    "bad",
    [
        "feature_nan",
        "feature_shape",
        "feature_dtype",
        "action_bounds",
        "action_rows",
        "examples",
        "zero_updates",
        "many_updates",
        "bool_counter",
        "policy",
    ],
)
def test_should_refuse_corrupt_restore_atomically(bad):
    model = learner()
    model.train_batch(*rows())
    before = model.snapshot_state()
    damaged = model.snapshot_state()
    if bad == "feature_nan":
        damaged.prototype_features[0, 0] = np.nan
    elif bad == "feature_shape":
        object.__setattr__(damaged, "prototype_features", np.zeros((2, 1)))
    elif bad == "feature_dtype":
        object.__setattr__(
            damaged, "prototype_features", damaged.prototype_features.astype(np.float32)
        )
    elif bad == "action_bounds":
        damaged.prototype_actions[0, 0] = -3.0
    elif bad == "action_rows":
        object.__setattr__(damaged, "prototype_actions", damaged.prototype_actions[:1])
    elif bad == "policy":
        object.__setattr__(damaged, "policy", policy(capacity=2))
    else:
        field, value = {
            "examples": ("training_examples", 3),
            "zero_updates": ("completed_updates", 0),
            "many_updates": ("completed_updates", 3),
            "bool_counter": ("completed_updates", True),
        }[bad]
        object.__setattr__(damaged, field, value)
    with pytest.raises(ValueError):
        model.restore_state(damaged)
    assert_same_state(before, model.snapshot_state())


@pytest.fixture
def native_boundary(monkeypatch, tmp_path):
    environment = FakeEnvironment()
    creations = []
    monkeypatch.setattr(port, "_make_environment", lambda: (creations.append(True), environment)[1])
    monkeypatch.setattr(smoke, "version", lambda _: "fake-native-boundary")
    output = tmp_path / "nearest-trace.json"
    monkeypatch.setattr(sys, "argv", ["smoke", "--output", str(output)])
    return environment, creations, output


def native_learner(*, cold=False):
    declared = ContinuousActionSpec(3, port.ACTION_UNITS, (-1.0,) * 4, (1.0,) * 4)
    transform = ObservationStandardization((0.0,) * 3, (1.0,) * 3, 2)
    model = adapter.NumpyNearestContinuousBCLearner(
        adapter.NearestContinuousBCPolicy(declared, transform)
    )
    if not cold:
        model.train_batch(
            np.array([[0.2, -0.3, 0.4], [1.2, -0.3, 0.4]]),
            np.array([[0.123456789, -1.0, 1.0, 0.0], [-0.333333333, 0.5, 0.0, -1.0]]),
        )
    return model


def read_receipt(output):
    return json.loads(output.read_text(encoding="utf8"))


def test_should_refuse_cold_nearest_model_before_native_creation(native_boundary):
    environment, creations, output = native_boundary
    assert smoke.main(learner=native_learner(cold=True)) == 1
    result = read_receipt(output)
    assert creations == [] and environment.commands == []
    assert result["creation_attempts"] == 0 and result["prediction_attempts"] == 0
    assert result["complete"] is False


def test_should_send_actual_float32_nearest_actions_and_preserve_complete_frozen_state(
    native_boundary,
):
    environment, creations, output = native_boundary
    model = native_learner()
    before = model.snapshot_state()
    assert smoke.main(learner=model) == 0
    result = read_receipt(output)
    assert creations == [True] and environment.close_calls == 1
    for index, command in enumerate(environment.commands):
        expected = before.prototype_actions[index].astype(np.float32)
        np.testing.assert_array_equal(command[0], expected)
        assert result["episode"]["transitions"][index]["action"] == expected.tolist()
    assert result["complete"] is True and result["optimizer_updates"] == 0
    assert result["frozen_state_exact"] is True and result["original_owner_exact"] is True
    assert result["prediction_attempts"] == result["prediction_returns"] == 2
    assert result["nearest_prediction_attempts"] == result["nearest_prediction_returns"] == 2
    assert (
        result["nearest_distance_candidate_upper_bound"]
        == result["nearest_distance_candidates_completed"]
        == 4
    )
    assert (
        result["nearest_distance_feature_upper_bound"]
        == result["nearest_distance_feature_values_completed"]
        == 12
    )
    assert result["nearest_selected_indices"] == [0, 1]
    assert (
        result["nearest_prototype_rows"] == 2 and result["nearest_prototype_numeric_bytes"] == 112
    )
    assert result["policy"] == "frozen_nearest_continuous_bc"
    assert result["source_update_definition"] == "continuous_nearest_bc_added_pairs_v1"
    assert result["work"]["completed_control_physics_steps"] == 10
    assert_same_state(before, model.snapshot_state())


def test_should_preserve_nearest_failure_upper_bound_without_claiming_completed_work(
    native_boundary, monkeypatch
):
    environment, _, output = native_boundary

    def broken(*_):
        raise RuntimeError("nearest retrieval failed")

    monkeypatch.setattr(adapter.NumpyNearestContinuousBCLearner, "predict_with_work", broken)
    assert smoke.main(learner=native_learner()) == 1
    result = read_receipt(output)
    assert result["nearest_prediction_attempts"] == result["prediction_attempts"] == 1
    assert result["nearest_prediction_returns"] == result["prediction_returns"] == 0
    assert result["nearest_distance_candidate_upper_bound"] == 2
    assert result["nearest_distance_feature_upper_bound"] == 6
    assert (
        result["nearest_distance_candidates_completed"]
        == result["nearest_distance_feature_values_completed"]
        == 0
    )
    assert result["nearest_selected_indices"] == [] and result["work"]["step_attempts"] == 0
    assert environment.commands == [] and environment.close_calls == 1


@pytest.mark.parametrize(
    "bad",
    [
        "foreign",
        "candidates",
        "features",
        "bool_count",
        "indices_list",
        "foreign_index",
        "index_range",
        "action_dtype",
        "action_shape",
        "action_bounds",
    ],
)
def test_should_refuse_corrupted_nearest_report_before_control_and_preserve_returned_work(
    native_boundary, monkeypatch, bad
):
    environment, _, output = native_boundary
    report = core.NearestPrediction(np.zeros((1, 4), dtype=np.float64), (0,), 2, 6)
    if bad == "foreign":
        report = object()
    elif bad == "candidates":
        report = replace(report, distance_candidates=3)
    elif bad == "features":
        report = replace(report, distance_feature_values=7)
    elif bad == "bool_count":
        report = replace(report, distance_candidates=True)
    elif bad == "indices_list":
        report = replace(report, selected_indices=[0])
    elif bad == "foreign_index":
        report = replace(report, selected_indices=(np.int64(0),))
    elif bad == "index_range":
        report = replace(report, selected_indices=(2,))
    elif bad == "action_dtype":
        report = replace(report, actions=report.actions.astype(np.float32))
    elif bad == "action_shape":
        report = replace(report, actions=report.actions.ravel())
    else:
        report.actions[0, 0] = 1.00000001
    monkeypatch.setattr(
        adapter.NumpyNearestContinuousBCLearner, "predict_with_work", lambda *_: report
    )
    assert smoke.main(learner=native_learner()) == 1
    result = read_receipt(output)
    assert result["nearest_prediction_attempts"] == result["nearest_prediction_returns"] == 1
    assert result["prediction_returns"] == 1 and result["validated_prediction_rows"] == 0
    assert result["nearest_distance_candidate_upper_bound"] == 2
    completed = 2 if bad.startswith("action_") else 0
    assert result["nearest_distance_candidates_completed"] == completed
    assert environment.commands == [] and environment.close_calls == 1


@pytest.mark.parametrize("component", ["features", "actions", "updates", "stats", "corrupt_shape"])
def test_should_refuse_private_nearest_state_mutation_before_control(
    native_boundary, monkeypatch, component
):
    environment, _, output = native_boundary
    model = native_learner()
    before = model.snapshot_state()
    original = adapter.NumpyNearestContinuousBCLearner.predict_with_work

    def corrupt(child, features):
        result = original(child, features)
        if component == "features":
            child._state.prototype_features[0, 0] += 0.01
        elif component == "actions":
            child._state.prototype_actions[0, 0] += 0.01
        elif component == "updates":
            object.__setattr__(child._state, "completed_updates", 2)
        elif component == "stats":
            object.__setattr__(child._policy.observation_standardization, "means", (0.01, 0.0, 0.0))
        else:
            object.__setattr__(child._state, "prototype_features", np.zeros((1, 3)))
        return result

    monkeypatch.setattr(adapter.NumpyNearestContinuousBCLearner, "predict_with_work", corrupt)
    assert smoke.main(learner=model) == 1
    result = read_receipt(output)
    assert environment.commands == [] and environment.close_calls == 1
    assert result["frozen_state_exact"] is not True and result["complete"] is False
    assert result["nearest_prediction_returns"] == 1 and result["work"]["step_attempts"] == 0
    assert_same_state(before, model.snapshot_state())


def test_should_reobserve_full_nearest_state_after_cleanup_and_clear_stale_true(
    native_boundary, monkeypatch
):
    environment, _, output = native_boundary
    model = native_learner()
    original_close = environment.close

    def corrupt_owner_during_close():
        original_close()
        object.__setattr__(model._state, "training_examples", 3)

    monkeypatch.setattr(environment, "close", corrupt_owner_during_close)
    assert smoke.main(learner=model) == 1
    result = read_receipt(output)
    assert result["complete"] is False and result["state_error"]
    assert result["original_owner_exact"] is None
    assert len(result["episode"]["transitions"]) == 2 and environment.close_calls == 1


@pytest.mark.parametrize("bad", ["subclass", "probe_policy"])
def test_should_refuse_non_train_or_foreign_nearest_subclass_before_scene(native_boundary, bad):
    _, creations, output = native_boundary

    class ForeignLearner(adapter.NumpyNearestContinuousBCLearner):
        pass

    declared = native_learner().policy
    if bad == "subclass":
        invalid = ForeignLearner(declared)
    else:
        invalid = adapter.NumpyNearestContinuousBCLearner(
            replace(declared, training_role="probe_train")
        )
    invalid.train_batch(np.zeros((1, 3)), np.zeros((1, 4)))
    assert smoke.main(learner=invalid) == 1
    assert creations == [] and read_receipt(output)["prediction_attempts"] == 0


def test_should_reserve_exclusive_output_before_nearest_native_work(native_boundary, monkeypatch):
    environment, creations, output = native_boundary
    output.write_bytes(b"preserved prior evidence\n")
    monkeypatch.setattr(
        adapter.NumpyNearestContinuousBCLearner,
        "predict_with_work",
        lambda *_: pytest.fail("existing output reached nearest retrieval"),
    )
    with pytest.raises(FileExistsError):
        smoke.main(learner=native_learner())
    assert output.read_bytes() == b"preserved prior evidence\n"
    assert creations == [] and environment.commands == [] and environment.close_calls == 0


def test_should_not_fit_statistics_or_use_rng_during_construction_append_or_prediction(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("nearest learner attempted hidden fitting or randomness")

    monkeypatch.setattr(adapter.observations, "fit_observation_standardization", forbidden)
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    model = learner()
    model.train_batch(*rows())
    assert model.predict_with_work(np.array([[0.0, 0.0]])).selected_indices == (0,)
