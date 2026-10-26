"""Actual stable-actor serving through fake robot IO; no native skill evidence."""

from dataclasses import replace
import json
import sys

import numpy as np
import pytest

from test_maniskill_bc_smoke import FakeEnvironment, learner as mlp_learner
from scripts import maniskill_continuous_smoke as smoke
from src.adapters.numpy_nearest_continuous_bc import (
    NearestContinuousBCPolicy,
    NumpyNearestContinuousBCLearner,
)
from src.app.actor_shadow import ActorShadowRuntime, StableActor
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.actor_ports import VersionedPrediction, VersionedState
from src.core.continuous_demonstration import ContinuousActionSpec
from src.core.continuous_episode_clock import EpisodeClock
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.observation_standardization import ObservationStandardization
from src.infra import maniskill_continuous as port


def owner(*, dimension=3, role="train", cold=False, units=None, lower=None):
    spec = ContinuousActionSpec(
        dimension,
        port.ACTION_UNITS if units is None else units,
        (-1.0,) * 4 if lower is None else lower,
        (1.0,) * 4,
    )
    statistics = ObservationStandardization((0.0,) * dimension, (1.0,) * dimension, 2)
    result = NumpyNearestContinuousBCLearner(
        NearestContinuousBCPolicy(spec, statistics, training_role=role)
    )
    if not cold:
        rows = [[0.2, -0.3, 0.4], [1.2, -0.3, 0.4]]
        if dimension == 4:
            rows = [rows[0] + [0.0], rows[1] + [0.02]]
        result.train_batch(
            np.asarray(rows, dtype=np.float64),
            np.array(
                [
                    [0.123456789, -1.0, 1.0, 0.0],
                    [-0.333333333, 0.5, 0.0, -1.0],
                ],
                dtype=np.float64,
            ),
        )
    return result


def runtime():
    source = owner()
    clock = LogicalClock()
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=1), lambda: 0.0)
    value = ActorShadowRuntime(
        source,
        actor_version="serving-0",
        candidate_version="candidate-0",
        clock=clock,
        budget=budget,
        max_experiences=1,
        max_consolidations=0,
    )
    return source, value, clock, budget


def assert_same_state(first, second):
    assert first.policy == second.policy
    assert first.completed_updates == second.completed_updates
    assert first.training_examples == second.training_examples
    np.testing.assert_array_equal(first.prototype_features, second.prototype_features)
    np.testing.assert_array_equal(first.prototype_actions, second.prototype_actions)


@pytest.fixture
def boundary(monkeypatch, tmp_path):
    environment = FakeEnvironment()
    creations = []

    def factory():
        creations.append(True)
        return environment

    monkeypatch.setattr(port, "_make_environment", factory)
    monkeypatch.setattr(smoke, "version", lambda _: "fake-native-boundary")
    output = tmp_path / "actor-trace.json"
    monkeypatch.setattr(sys, "argv", ["smoke", "--output", str(output)])
    return environment, creations, output


def receipt(output):
    return json.loads(output.read_text(encoding="utf8"))


def test_should_call_actual_runtime_actor_without_any_new_learner_fork(boundary, monkeypatch):
    environment, creations, output = boundary
    source, shadow, _, _ = runtime()
    supplied = shadow.actor
    initial = supplied.snapshot_state()
    backend = supplied._learner
    calls, snapshots = [], []
    actual_predict, actual_snapshot = StableActor.predict, StableActor.snapshot_state

    def predict(actor, features):
        calls.append((actor, features.copy()))
        result = actual_predict(actor, features)
        assert type(result) is VersionedPrediction and result.actor_version == "serving-0"
        return result

    def snapshot(actor):
        snapshots.append(actor)
        return actual_snapshot(actor)

    monkeypatch.setattr(StableActor, "predict", predict)
    monkeypatch.setattr(StableActor, "snapshot_state", snapshot)
    monkeypatch.setattr(
        NumpyNearestContinuousBCLearner,
        "fork",
        lambda *_: pytest.fail("robot serving forked a replacement learner"),
    )
    assert smoke.main(actor=supplied) == 0
    result = receipt(output)
    assert creations == [True] and environment.close_calls == 1
    assert len(calls) == 2 and all(actual is supplied for actual, _ in calls)
    assert snapshots and all(actual is supplied for actual in snapshots)
    assert supplied._learner is backend
    for index, command in enumerate(environment.commands):
        expected = initial.state.prototype_actions[index].astype(np.float32)
        np.testing.assert_array_equal(command[0], expected)
        assert result["episode"]["transitions"][index]["action"] == expected.tolist()
    assert result["policy"] == "frozen_stable_nearest_continuous_bc"
    assert result["actor_version"] == "serving-0"
    assert result["actor_version_exact"] is result["actor_backend_exact"] is True
    assert result["frozen_state_exact"] is result["original_owner_exact"] is True
    assert result["actor_prediction_version_exact"] is True
    assert result["state_guard_scope"] == "actual_serving_actor_no_private_fork"
    assert result["actor_prediction_attempts"] == result["actor_prediction_returns"] == 2
    assert result["actor_prediction_versions"] == ["serving-0", "serving-0"]
    assert (
        result["actor_distance_count_source"]
        == "derived_from_exact_unchanged_typed_actor_and_query_shapes"
    )
    assert (
        result["actor_distance_candidate_upper_bound"]
        == result["actor_distance_candidates_completed"]
        == 4
    )
    assert (
        result["actor_distance_feature_upper_bound"]
        == result["actor_distance_feature_values_completed"]
        == 12
    )
    assert result["validated_prediction_rows"] == 2 and result["validated_prediction_bytes"] == 64
    assert result["source_updates"] == 1 and result["source_training_rows"] == 2
    assert (
        result["nearest_prototype_rows"] == 2 and result["nearest_prototype_numeric_bytes"] == 112
    )
    assert result["complete"] is True and result["optimizer_updates"] == 0
    assert result["nearest_prediction_attempts"] == result["nearest_prediction_returns"] == 0
    assert result["nearest_selected_indices"] == []
    assert_same_state(initial.state, supplied.snapshot_state().state)
    assert_same_state(initial.state, source.snapshot_state())


def test_should_keep_actual_actor_warm_while_delayed_candidate_appends_during_control(
    boundary, monkeypatch
):
    environment, _, output = boundary
    source, shadow, clock, budget = runtime()
    before = source.snapshot_state()
    features = np.array([[10.2, -0.3, 0.4], [11.2, -0.3, 0.4]])
    actions = np.array([[0.9, 0.0, 0.0, 1.0], [0.8, 0.0, 0.0, 1.0]])
    shadow.record_experience(
        Experience(
            "B", "train-B", 0, "serving-0", features, "train", ExperiencePermissions(training=True)
        )
    )
    shadow.record_label(LabelArrival("label-B", "B", "train-B", 2, "serving-0", actions))
    assert shadow.train_ready() == () and budget.updates_completed == 0
    native_step = environment.step

    def step(command):
        result = native_step(command)
        if len(environment.commands) == 1:
            clock.advance_to(2)
            assert len(shadow.train_ready()) == 1
        return result

    monkeypatch.setattr(environment, "step", step)
    assert smoke.main(actor=shadow.actor, train_seed=19) == 0
    result = receipt(output)
    assert environment.reset_calls == [19] and result["context"] == "pick-cube-train-19"
    assert budget.updates_completed == 1 and shadow.train_ready() == ()
    assert shadow.candidate_snapshot().state.training_examples == 4
    assert shadow.candidate_snapshot().state.completed_updates == 2
    assert_same_state(before, source.snapshot_state())
    assert_same_state(before, shadow.actor.snapshot_state().state)
    for index, command in enumerate(environment.commands):
        np.testing.assert_array_equal(
            command[0], before.prototype_actions[index].astype(np.float32)
        )
    assert result["actor_version"] == "serving-0" and result["frozen_state_exact"] is True


@pytest.mark.parametrize(
    "bad",
    [
        "foreign_actor",
        "actor_subclass",
        "mlp_backend",
        "cold",
        "probe_role",
        "units",
        "bounds",
        "version",
        "corrupt",
    ],
)
def test_should_refuse_invalid_stable_actor_before_native_creation(boundary, bad):
    _, creations, output = boundary
    if bad == "foreign_actor":
        supplied = object()
    elif bad == "actor_subclass":

        class ForeignActor(StableActor):
            pass

        supplied = ForeignActor(owner(), version="serving-0")
    elif bad == "mlp_backend":
        supplied = StableActor(mlp_learner(), version="serving-0")
    else:
        options = {
            "cold": dict(cold=True),
            "probe_role": dict(role="probe_train"),
            "units": dict(units=("m",) * 4),
            "bounds": dict(lower=(-2.0,) * 4),
        }.get(bad, {})
        supplied = StableActor(owner(**options), version="serving-0")
        if bad == "version":
            supplied._version = " bad-version "
        elif bad == "corrupt":
            supplied._learner._state.prototype_features[0, 0] = np.nan
    assert smoke.main(actor=supplied) == 1
    result = receipt(output)
    assert creations == [] and result["creation_attempts"] == result["creation_returns"] == 0
    assert result["prediction_attempts"] == 0 and result["complete"] is False


def test_should_refuse_actor_and_learner_together_before_public_snapshot_or_scene(
    boundary, monkeypatch
):
    _, creations, output = boundary
    supplied = StableActor(owner(), version="serving-0")
    learner = owner()
    monkeypatch.setattr(
        StableActor,
        "snapshot_state",
        lambda *_: pytest.fail("ambiguous sources reached actor observation"),
    )
    assert smoke.main(actor=supplied, learner=learner) == 1
    assert creations == [] and receipt(output)["prediction_attempts"] == 0


@pytest.mark.parametrize(
    "bad", ["foreign", "wrong_version", "version_type", "float32", "shape", "nan", "bounds"]
)
def test_should_refuse_corrupt_versioned_prediction_before_native_control(
    boundary, monkeypatch, bad
):
    environment, _, output = boundary
    supplied = StableActor(owner(), version="serving-0")
    original = StableActor.predict

    def corrupt(actor, features):
        result = original(actor, features)
        if bad == "foreign":
            return object()
        if bad == "wrong_version":
            return replace(result, actor_version="other")
        if bad == "version_type":
            return replace(result, actor_version=np.str_("serving-0"))
        if bad == "float32":
            return replace(result, prediction=result.prediction.astype(np.float32))
        if bad == "shape":
            return replace(result, prediction=result.prediction.ravel())
        result.prediction[0, 0] = np.nan if bad == "nan" else 1.00000001
        return result

    monkeypatch.setattr(StableActor, "predict", corrupt)
    assert smoke.main(actor=supplied) == 1
    result = receipt(output)
    assert environment.commands == [] and environment.close_calls == 1
    assert result["prediction_attempts"] == result["prediction_returns"] == 1
    assert result["validated_prediction_rows"] == 0 and result["work"]["step_attempts"] == 0
    assert result["complete"] is False


@pytest.mark.parametrize(
    "component", ["version", "features", "actions", "updates", "statistics", "backend"]
)
def test_should_refuse_actor_identity_version_or_full_state_mutation_before_control(
    boundary, monkeypatch, component
):
    environment, _, output = boundary
    supplied = StableActor(owner(), version="serving-0")
    replacement = owner()
    original = StableActor.predict

    def corrupt(actor, features):
        result = original(actor, features)
        if component == "version":
            actor._version = "serving-1"
        elif component == "features":
            actor._learner._state.prototype_features[0, 0] += 0.01
        elif component == "actions":
            actor._learner._state.prototype_actions[0, 0] += 0.01
        elif component == "updates":
            object.__setattr__(actor._learner._state, "completed_updates", 2)
        elif component == "statistics":
            object.__setattr__(
                actor._learner._policy.observation_standardization, "means", (0.01, 0.0, 0.0)
            )
        else:
            actor._learner = replacement
        return result

    monkeypatch.setattr(StableActor, "predict", corrupt)
    assert smoke.main(actor=supplied) == 1
    result = receipt(output)
    assert environment.commands == [] and environment.close_calls == 1
    assert result["complete"] is False and result["work"]["step_attempts"] == 0
    if component == "backend":
        assert result["actor_backend_exact"] is False
    elif component == "version":
        assert result["actor_version_exact"] is False
    else:
        assert result["frozen_state_exact"] is not True


def test_should_preserve_actor_prediction_exception_without_admitting_control(
    boundary, monkeypatch
):
    environment, _, output = boundary
    supplied = StableActor(owner(), version="serving-0")

    def broken(*_):
        raise RuntimeError("actor prediction did not return")

    monkeypatch.setattr(StableActor, "predict", broken)
    assert smoke.main(actor=supplied) == 1
    result = receipt(output)
    assert result["prediction_attempts"] == 1 and result["prediction_returns"] == 0
    assert result["actor_prediction_attempts"] == 1 and result["actor_prediction_returns"] == 0
    assert result["actor_distance_candidate_upper_bound"] == 2
    assert result["actor_distance_feature_upper_bound"] == 6
    assert (
        result["actor_distance_candidates_completed"]
        == result["actor_distance_feature_values_completed"]
        == 0
    )
    assert (
        result["actor_prediction_versions"] == []
        and result["actor_prediction_version_exact"] is None
    )
    assert result["nearest_prediction_attempts"] == result["nearest_prediction_returns"] == 0
    assert (
        result["nearest_distance_candidates_completed"]
        == result["nearest_distance_feature_values_completed"]
        == 0
    )
    assert result["nearest_selected_indices"] == []
    assert result["validated_prediction_rows"] == 0 and environment.commands == []
    assert environment.close_calls == 1 and "did not return" in result["error"]


def test_should_recheck_actual_actor_after_control_before_serving_next_step(boundary, monkeypatch):
    environment, _, output = boundary
    supplied = StableActor(owner(), version="serving-0")
    original = environment.step

    def mutate_after_control(command):
        result = original(command)
        supplied._learner._state.prototype_actions[0, 0] += 0.01
        return result

    monkeypatch.setattr(environment, "step", mutate_after_control)
    assert smoke.main(actor=supplied) == 1
    result = receipt(output)
    assert len(environment.commands) == result["prediction_attempts"] == 1
    assert result["work"]["completed_steps"] == 1 and result["frozen_state_exact"] is False


def test_should_clear_stale_exact_flags_when_cleanup_makes_state_unobservable(
    boundary, monkeypatch
):
    environment, _, output = boundary
    supplied = StableActor(owner(), version="serving-0")
    original = environment.close

    def corrupt_during_cleanup():
        original()
        object.__setattr__(supplied._learner._state, "training_examples", 3)

    monkeypatch.setattr(environment, "close", corrupt_during_cleanup)
    assert smoke.main(actor=supplied) == 1
    result = receipt(output)
    assert result["complete"] is False and result["state_error"]
    assert result["frozen_state_exact"] is None and result["original_owner_exact"] is None
    assert len(result["episode"]["transitions"]) == 2 and environment.close_calls == 1


@pytest.mark.parametrize("bad", ["bool_counter", "float32_prototypes", "foreign_versioned_state"])
def test_should_revalidate_late_public_snapshot_instead_of_accepting_equal_values(
    boundary, monkeypatch, bad
):
    environment, _, output = boundary
    supplied = StableActor(owner(), version="serving-0")
    before = supplied.snapshot_state()
    original = StableActor.snapshot_state

    class ForeignState(VersionedState):
        pass

    if bad == "float32_prototypes":
        # Exactly representable values ensure equality alone cannot catch dtype loss.
        supplied._learner._state.prototype_actions[:] = [
            [0.0, -1.0, 1.0, 0.0],
            [0.5, 0.5, 0.0, -1.0],
        ]
        before = supplied.snapshot_state()

    def corrupt_snapshot(actor):
        tagged = original(actor)
        if environment.commands:
            if bad == "bool_counter":
                object.__setattr__(tagged.state, "completed_updates", True)
            elif bad == "float32_prototypes":
                object.__setattr__(
                    tagged.state,
                    "prototype_actions",
                    tagged.state.prototype_actions.astype(np.float32),
                )
            else:
                return ForeignState(tagged.actor_version, tagged.state)
        return tagged

    monkeypatch.setattr(StableActor, "snapshot_state", corrupt_snapshot)
    assert smoke.main(actor=supplied) == 1
    result = receipt(output)
    assert len(environment.commands) == 1 and result["prediction_attempts"] == 1
    assert result["frozen_state_exact"] is None and result["original_owner_exact"] is None
    assert result["complete"] is False and result["state_error"]
    assert_same_state(before.state, original(supplied).state)


def test_should_clear_previous_valid_prediction_tag_when_next_tag_is_wrong(boundary, monkeypatch):
    environment, _, output = boundary
    supplied = StableActor(owner(), version="serving-0")
    original = StableActor.predict
    calls = []

    def corrupt_second(actor, features):
        result = original(actor, features)
        calls.append(True)
        return result if len(calls) == 1 else replace(result, actor_version="wrong-second-tag")

    monkeypatch.setattr(StableActor, "predict", corrupt_second)
    assert smoke.main(actor=supplied) == 1
    result = receipt(output)
    assert len(environment.commands) == 1 and result["prediction_attempts"] == 2
    assert result["actor_prediction_version_exact"] is False
    assert result["actor_prediction_versions"] == ["serving-0"]
    assert result["actor_distance_candidate_upper_bound"] == 4
    assert result["actor_distance_candidates_completed"] == 2


def test_should_encode_existing_clock_recipe_for_actual_actor_and_preserve_raw_trace(
    boundary, monkeypatch
):
    environment, _, output = boundary
    supplied = StableActor(owner(dimension=4), version="serving-0")
    original = StableActor.predict
    observations = []

    def capture(actor, features):
        observations.append(features.copy())
        return original(actor, features)

    monkeypatch.setattr(StableActor, "predict", capture)
    assert smoke.main(actor=supplied, observation_clock=EpisodeClock(50, True)) == 0
    result = receipt(output)
    assert [row[0, -1] for row in observations] == [0.0, 0.02]
    assert result["clock_values"] == [0.0, 0.02] and result["clock_configuration_exact"] is True
    assert result["episode"]["spec"]["observation_dim"] == 3
    assert len(environment.commands) == 2 and result["frozen_state_exact"] is True


def test_should_refuse_postcreation_actor_observation_spec_mismatch_before_prediction(
    boundary, monkeypatch
):
    environment, creations, output = boundary
    supplied = StableActor(owner(dimension=4), version="serving-0")
    monkeypatch.setattr(
        StableActor, "predict", lambda *_: pytest.fail("mismatched actor spec reached serving")
    )
    assert smoke.main(actor=supplied) == 1
    assert creations == [True] and environment.commands == [] and environment.close_calls == 1
    assert receipt(output)["prediction_attempts"] == 0


def test_should_refuse_invalid_clock_before_actor_observation_or_creation(boundary, monkeypatch):
    _, creations, output = boundary
    supplied = StableActor(owner(dimension=4), version="serving-0")
    monkeypatch.setattr(
        StableActor,
        "snapshot_state",
        lambda *_: pytest.fail("invalid clock reached actor observation"),
    )
    assert smoke.main(actor=supplied, observation_clock=EpisodeClock(49, True)) == 1
    result = receipt(output)
    assert creations == [] and result["actor_prediction_attempts"] == 0


def test_should_reserve_exclusive_output_before_observing_or_serving_actor(boundary, monkeypatch):
    environment, creations, output = boundary
    supplied = StableActor(owner(), version="serving-0")
    output.write_bytes(b"preserved failed attempt\n")
    monkeypatch.setattr(
        StableActor,
        "snapshot_state",
        lambda *_: pytest.fail("existing output admitted actor snapshot"),
    )
    with pytest.raises(FileExistsError):
        smoke.main(actor=supplied)
    assert output.read_bytes() == b"preserved failed attempt\n"
    assert creations == [] and environment.commands == [] and environment.close_calls == 0
