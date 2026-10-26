"""Fixed-input nearest memory and causal native taskspace serving.

Prospective per platform: three nearest initializers, four successful appends /
seven rows, seven prediction rows /15 candidates /90 distance feature values,
two fake scenes /three controls. No empirical fit, neural RNG, SGD or physics.
Only module fixtures perform successful append/prediction work; ownership and
refusal checks use their owned snapshots and RNG-free forks.
"""

from dataclasses import asdict, fields, replace
from typing import Any

import numpy as np
import pytest

from scripts import maniskill_continuous_smoke as smoke
from src.adapters import numpy_nearest_continuous_bc as adapter
from src.app.actor_shadow import StableActor
from src.core import pick_cube_action_history as history
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)
from src.core.observation_standardization import ObservationStandardization
from src.core.pick_cube_taskspace_history import PickCubeTaskSpaceHistoryRecipe
from src.infra import maniskill_continuous as native
from test_maniskill_bc_smoke import FakeEnvironment


RECIPE = PickCubeTaskSpaceHistoryRecipe()


def spec(dimensions=2):
    return ContinuousActionSpec(dimensions, native.ACTION_UNITS, (-1.0,) * 4, (1.0,) * 4)


def raw(index):
    values = [0.0] * 42
    values[0], values[18] = float(index), float(index > 0)
    values[22] = values[32] = 1.0
    values[36:42] = (0.05 + index * 0.025, -0.025, 0.0125, -0.1, 0.025, 0.05)
    return tuple(values)


def episode(role="train"):
    actions = ((0.5, -0.25, 0.0, -1.0), (-0.5, 0.25, 0.0, 1.0))
    return ContinuousEpisode(
        "memory-source",
        "pick-cube",
        role,
        spec(42),
        tuple(
            ContinuousTransition(i, raw(i), action, float(i == 1), raw(i + 1), i == 1, False)
            for i, action in enumerate(actions)
        ),
    )


class TaskspaceEnvironment(FakeEnvironment):
    def reset(self, *, seed):
        _, info = super().reset(seed=seed)
        return np.asarray([raw(0)], dtype=np.float64), info

    def step(self, command):
        returned = super().step(command)
        return np.asarray([raw(len(self.commands))], dtype=np.float64), *returned[1:]


def assert_state(first, second):
    assert first.policy == second.policy and type(first.policy) is type(second.policy)
    assert first.completed_updates == second.completed_updates
    assert first.training_examples == second.training_examples
    np.testing.assert_array_equal(first.prototype_features, second.prototype_features)
    np.testing.assert_array_equal(first.prototype_actions, second.prototype_actions)


@pytest.fixture(scope="module")
def memories():
    features = np.array([[10.0, 0.0], [0.0, 2.0]], dtype=np.float64)
    actions = np.array([[0.5, 0.0, 0.0, -1.0], [-0.5, 0.0, 0.0, 1.0]], dtype=np.float64)
    fixed = adapter.NumpyNearestContinuousBCLearner(
        adapter.FixedInputNearestContinuousBCPolicy(spec(), max_prototypes=3)
    )
    fitted = adapter.NumpyNearestContinuousBCLearner(
        adapter.NearestContinuousBCPolicy(
            spec(), ObservationStandardization((0.0, 0.0), (10.0, 1.0), 2)
        )
    )
    fixed_update = fixed.train_batch(features, actions)
    fitted.train_batch(features, actions)
    source = episode()
    model_episode = history.action_history_training_episode(source, RECIPE)
    native_model = adapter.NumpyNearestContinuousBCLearner(
        adapter.FixedInputNearestContinuousBCPolicy(model_episode.spec)
    )
    native_model.train_episode(model_episode)
    fixed_prediction = fixed.predict_with_work(np.array([[0.0, 0.0], [5.0, 1.0]]))
    fitted_prediction = fitted.predict_with_work(np.array([[0.0, 0.0]]))
    appended = fixed.fork()
    appended.train_batch(features[:1], np.array([[-0.75, 0.0, 0.0, 1.0]]))
    tie = appended.predict_with_work(features[:1])
    return dict(
        fixed=fixed,
        fitted=fitted,
        native=native_model,
        source=source,
        model_episode=model_episode,
        fixed_update=fixed_update,
        fixed_prediction=fixed_prediction,
        fitted_prediction=fitted_prediction,
        appended=appended,
        tie=tie,
        features=features.copy(),
        actions=actions.copy(),
    )


def test_should_select_identity_without_inventing_fitted_rows_or_changing_policy_fields():
    policy = adapter.FixedInputNearestContinuousBCPolicy(spec())
    assert isinstance(policy, adapter.NearestContinuousBCPolicy)
    assert policy.observation_standardization is None
    assert policy.max_prototypes == 256 and policy.training_role == "train"
    assert [field.name for field in fields(policy)] == [
        "spec",
        "observation_standardization",
        "max_prototypes",
        "training_role",
    ]
    assert asdict(policy)["observation_standardization"] is None


def test_should_keep_legacy_fitted_mode_strict_when_statistics_are_absent():
    with pytest.raises(ValueError):
        adapter.NearestContinuousBCPolicy(spec(), None)


def test_should_refuse_fitted_statistics_on_the_explicit_identity_variant():
    with pytest.raises(ValueError):
        adapter.FixedInputNearestContinuousBCPolicy(
            spec(), ObservationStandardization((0.0, 0.0), (1.0, 1.0), 2)
        )


@pytest.mark.parametrize("capacity", [0, True])
def test_should_validate_fixed_capacity_before_allocating_a_memory(capacity):
    with pytest.raises(ValueError):
        adapter.FixedInputNearestContinuousBCPolicy(spec(), max_prototypes=capacity)


def test_should_store_identity_coordinates_and_use_hand_computed_squared_distance(memories):
    state = memories["fixed"].snapshot_state()
    np.testing.assert_array_equal(state.prototype_features, memories["features"])
    assert memories["fixed_update"].definition == adapter.OBJECTIVE
    assert memories["fixed_update"].value == 2
    result = memories["fixed_prediction"]
    assert result.selected_indices == (1, 0)
    assert result.distance_candidates == 4 and result.distance_feature_values == 8
    np.testing.assert_array_equal(result.actions, memories["actions"][[1, 0]])


def test_should_preserve_existing_fitted_geometry_instead_of_silently_replacing_it(memories):
    fitted = memories["fitted"].snapshot_state()
    assert type(fitted.policy) is adapter.NearestContinuousBCPolicy
    np.testing.assert_array_equal(fitted.prototype_features, [[1.0, 0.0], [0.0, 2.0]])
    assert memories["fitted_prediction"].selected_indices == (0,)
    assert memories["fitted_prediction"].distance_feature_values == 4
    assert memories["fixed_prediction"].selected_indices[0] == 1


def test_should_own_snapshots_and_preserve_the_exact_variant_through_fork_restore(memories):
    owner = memories["fixed"]
    before = owner.snapshot_state()
    child = owner.fork()
    child.restore_state(before)
    assert type(child.policy) is adapter.FixedInputNearestContinuousBCPolicy
    assert_state(before, child.snapshot_state())
    detached = owner.snapshot_state()
    detached.prototype_features.fill(99.0)
    detached.prototype_actions.fill(0.0)
    external_policy = owner.policy
    object.__setattr__(external_policy, "max_prototypes", 1)
    assert_state(before, owner.snapshot_state())
    assert_state(before, child.snapshot_state())


def test_should_append_without_overwriting_an_earlier_conflicting_exact_input_and_stop_at_capacity(
    memories, monkeypatch
):
    owner = memories["appended"]
    before = owner.snapshot_state()
    assert before.completed_updates == 2 and before.training_examples == 3
    assert memories["tie"].selected_indices == (0,)
    assert memories["tie"].distance_candidates == 3 and memories["tie"].distance_feature_values == 6
    np.testing.assert_array_equal(memories["tie"].actions, memories["actions"][:1])
    np.testing.assert_array_equal(before.prototype_actions[-1], [-0.75, 0.0, 0.0, 1.0])

    def forbidden(*args):
        pytest.fail("capacity refusal reached normalization")

    with monkeypatch.context() as patch:
        patch.setattr(adapter.observations, "standardize_observations", forbidden)
        with pytest.raises(ValueError):
            owner.train_batch(memories["features"][:1], memories["actions"][:1])
    assert_state(before, owner.snapshot_state())


@pytest.mark.parametrize("role", ["probe_train", "probe_test", "final_test"])
def test_should_refuse_other_episode_roles_before_normalization_or_append(
    memories, monkeypatch, role
):
    owner = memories["native"].fork()
    before = owner.snapshot_state()

    def forbidden(*args):
        pytest.fail("role refusal admitted normalization")

    monkeypatch.setattr(adapter.observations, "standardize_observations", forbidden)
    with pytest.raises(ValueError):
        owner.train_episode(replace(memories["model_episode"], role=role))
    assert_state(before, owner.snapshot_state())


def test_should_refuse_a_foreign_policy_subclass_before_any_learner_construction():
    class Foreign(adapter.FixedInputNearestContinuousBCPolicy):
        pass

    foreign = object.__new__(Foreign)
    original = adapter.FixedInputNearestContinuousBCPolicy(spec())
    for field in fields(original):
        object.__setattr__(foreign, field.name, getattr(original, field.name))
    with pytest.raises(ValueError):
        adapter._policy(foreign)


def test_should_refuse_cross_variant_restore_atomically_even_when_the_specs_match(memories):
    child = memories["fixed"].fork()
    before = child.snapshot_state()
    with pytest.raises(ValueError):
        child.restore_state(memories["fitted"].snapshot_state())
    assert_state(before, child.snapshot_state())


@pytest.fixture(scope="module")
def serving(memories, tmp_path_factory):
    records = {}
    original_predict = adapter.NumpyNearestContinuousBCLearner.predict_with_work
    observation_property = vars(native.ManiSkillPickCubeEpisode)["observation"]
    assert type(observation_property) is property
    assert observation_property.fget is not None
    get_observation = observation_property.fget
    for late in (False, True):
        environment = TaskspaceEnvironment()
        captured: dict[str, Any] = dict(inputs=[], inference=None, mutations=0)
        source = memories["native"].fork()
        before = source.snapshot_state()

        def predicting(owner, features):
            captured["inference"] = owner
            captured["inputs"].append(features.copy())
            return original_predict(owner, features)

        def observation(port):
            value = get_observation(port)
            if late and port._records and captured["mutations"] == 0:
                captured["inference"]._state.prototype_actions[0, 0] = -0.75
                captured["mutations"] += 1
            return value

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(native, "_make_environment", lambda: environment)
            patch.setattr(adapter.NumpyNearestContinuousBCLearner, "predict_with_work", predicting)
            patch.setattr(native.ManiSkillPickCubeEpisode, "observation", property(observation))
            patch.setattr(smoke, "version", lambda _: "fake-fixed-memory")
            outcome = smoke.run_frozen_rollout(
                tmp_path_factory.mktemp("fixed-memory") / "trace.json",
                learner=source,
                action_history_recipe=RECIPE,
            )
        assert_state(before, source.snapshot_state())
        records[late] = outcome, captured, environment, before
    return records


def test_should_predict_from_raw42_and_previous_executed_actions_without_changing_frozen_memory(
    serving, memories
):
    outcome, captured, environment, before = serving[False]
    report = outcome.report
    assert report["complete"] is True, report
    assert report["policy"] == "frozen_nearest_continuous_bc"
    assert report["source_updates"] == 1 and report["source_training_rows"] == 2
    assert report["frozen_state_exact"] is report["original_owner_exact"] is True
    assert report["nearest_prediction_attempts"] == report["nearest_prediction_returns"] == 2
    assert report["nearest_distance_candidates_completed"] == 4
    assert report["nearest_distance_feature_values_completed"] == 48
    assert report["nearest_selected_indices"] == [0, 1]
    assert report["optimizer_updates"] == 0 and outcome.corrections is None
    assert report["action_history_exact"] is True
    assert report["action_history"]["recipe"] == asdict(RECIPE)
    assert report["action_history"]["work"]["committed_controls"] == 2
    assert report["action_history"]["work"]["source_values_validated"] == 84
    for features, expected in zip(
        captured["inputs"], memories["model_episode"].transitions, strict=True
    ):
        np.testing.assert_array_equal(features, [expected.observation])
    assert tuple(captured["inputs"][0][0][-5:]) == (0.0,) * 5
    assert tuple(captured["inputs"][1][0][-5:]) == (
        *memories["source"].transitions[0].action,
        1 / 6,
    )
    raw_episode = report["episode"]
    assert raw_episode["spec"]["observation_dim"] == 42
    assert [tuple(row["action"]) for row in raw_episode["transitions"]] == [
        row.action for row in memories["source"].transitions
    ]
    assert (
        len(environment.commands) == 2
        and environment.reset_calls == [17]
        and environment.close_calls == 1
    )
    assert_state(before, captured["inference"].snapshot_state())


def test_should_refuse_late_observation_getter_mutation_before_a_second_prediction_or_control(
    serving,
):
    outcome, captured, environment, _ = serving[True]
    report = outcome.report
    assert report["complete"] is False and report["error"] is not None
    assert captured["mutations"] == 1 and len(captured["inputs"]) == 1
    assert report["nearest_prediction_returns"] == 1
    assert report["nearest_distance_feature_values_completed"] == 24
    assert report["frozen_state_exact"] is False and report["original_owner_exact"] is True
    assert report["work"]["step_attempts"] == report["work"]["completed_steps"] == 1
    assert report["action_history"]["work"]["committed_controls"] == 1
    assert len(environment.commands) == 1 and environment.close_calls == 1
    assert outcome.corrections is None


@pytest.mark.parametrize(
    "fault",
    [
        "history47",
        "zero",
        "queries",
        "clock",
        "visual",
        "probe_role",
        "fitted",
        "missing_recipe",
        "actor",
    ],
)
def test_should_refuse_unsupported_fixed_memory_serving_before_environment_creation(
    memories, monkeypatch, tmp_path, fault
):
    owner = memories["native"].fork()
    options: dict[str, Any] = dict(learner=owner, action_history_recipe=RECIPE)
    if fault == "history47":
        options["action_history_recipe"] = history.PickCubeActionHistoryRecipe()
    elif fault == "zero":
        options["action_history_recipe"] = PickCubeTaskSpaceHistoryRecipe(mode="zero")
    elif fault == "queries":
        options["corrective_queries"] = True
    elif fault == "clock":
        options["observation_clock"] = object()
    elif fault == "visual":
        options["visual_observation"] = True
    elif fault == "missing_recipe":
        options.pop("action_history_recipe")
    elif fault == "actor":
        options.pop("learner")
        options["actor"] = StableActor(owner, version="fixed-memory")
    else:
        snapshot = owner.snapshot_state()
        policy = (
            replace(owner.policy, training_role="probe_train")
            if fault == "probe_role"
            else adapter.NearestContinuousBCPolicy(
                snapshot.policy.spec, ObservationStandardization((0.0,) * 12, (1.0,) * 12, 2)
            )
        )
        owner._policy = policy
        owner.restore_state(replace(snapshot, policy=policy))

    def forbidden(*args, **kwargs):
        pytest.fail("unsupported memory serving created an environment")

    monkeypatch.setattr(native, "_make_environment", forbidden)
    outcome = smoke.run_frozen_rollout(tmp_path / "refused.json", **options)
    assert outcome.report["complete"] is False and outcome.report["error"] is not None
    assert outcome.report["creation_attempts"] == 0
    assert outcome.report["prediction_attempts"] == 0
