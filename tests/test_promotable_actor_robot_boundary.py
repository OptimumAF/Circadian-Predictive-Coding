"""Actual promotable serving under fake robot IO; synthetic guards only."""

from dataclasses import replace
from hashlib import sha256
import json
import pickle
import sys

import numpy as np
import pytest

from test_maniskill_bc_smoke import FakeEnvironment
from scripts import maniskill_continuous_smoke as smoke
from src.adapters.numpy_nearest_continuous_bc import (
    NearestContinuousBCPolicy,
    NumpyNearestContinuousBCLearner,
)
from src.app.actor_shadow import ActorShadowRuntime
from src.app.promotion_guard_evaluation import PromotionGuardEvaluator
from src.app.serving_promotion import PromotableActor, ServingPromotionController
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.actor_ports import VersionedPrediction
from src.core.continuous_demonstration import ContinuousActionSpec
from src.core.continuous_episode_clock import EpisodeClock
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.observation_standardization import ObservationStandardization
from src.core.promotion_guard import GuardBatch, PromotionPolicy
from src.core.serving_ports import CachedPrediction, ServingConfiguration
from src.infra import maniskill_continuous as port


def digest(value):
    return sha256(pickle.dumps(value)).hexdigest()


def model(*, dimension=3, role="train", cold=False):
    spec = ContinuousActionSpec(dimension, port.ACTION_UNITS, (-1.0,) * 4, (1.0,) * 4)
    statistics = ObservationStandardization((0.0,) * dimension, (1.0,) * dimension, 2)
    owner = NumpyNearestContinuousBCLearner(
        NearestContinuousBCPolicy(spec, statistics, training_role=role)
    )
    if not cold:
        features = [[10.2, -0.3, 0.4], [11.2, -0.3, 0.4]]
        if dimension == 4:
            features = [row + [0.0] for row in features]
        owner.train_batch(
            np.asarray(features, dtype=np.float64),
            np.asarray([[-0.5, 0.0, 0.0, -1.0]] * 2, dtype=np.float64),
        )
    return owner


def actor_for(source=None, *, metadata=None):
    return PromotableActor(
        model() if source is None else source,
        version="serving-0",
        configuration=ServingConfiguration(3, 2),
        feature_digest=digest,
        metadata={"route": ["warm"]} if metadata is None else metadata,
    )


def promotion_fixture():
    """Guard identities are synthetic, disjoint and never relabeled TRAIN rows."""
    source = model()
    actor = actor_for(source)
    clock = LogicalClock()
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=1), lambda: 0.0)
    runtime = ActorShadowRuntime(
        source,
        actor=actor,
        actor_version="serving-0",
        candidate_version="candidate-0",
        clock=clock,
        budget=budget,
        max_experiences=1,
        max_consolidations=0,
    )
    features = np.asarray([[0.2, -0.3, 0.4], [1.2, -0.3, 0.4]], dtype=np.float64)
    targets = np.asarray([[0.75, 0.0, 0.0, 1.0]] * 2, dtype=np.float64)
    runtime.record_experience(
        Experience(
            "sample",
            "synthetic-train",
            0,
            "serving-0",
            features,
            "train",
            ExperiencePermissions(training=True),
        )
    )

    def build(state):
        owner = NumpyNearestContinuousBCLearner(state.policy)
        owner.restore_state(state)
        return owner

    def valid_state(snapshot):
        try:
            build(snapshot)
            return True
        except ValueError:
            return False

    declared = PromotionPolicy(
        "synthetic_negative_mse_v1",
        "prototype_numeric_bytes_v1",
        0.0,
        0.1,
        0.0,
        1.0,
        1024**2,
        ("bounded",),
    )
    evaluator = PromotionGuardEvaluator(
        build_learner=build,
        utility=lambda prediction, truth: -float(np.mean((prediction - truth) ** 2)),
        actions=lambda prediction: (
            ("bounded",) if bool(np.all(np.abs(prediction) <= 1)) else ("unsafe",)
        ),
        state_valid=valid_state,
        prediction_valid=lambda prediction: bool(np.isfinite(prediction).all()),
        resource_bytes=lambda state: (
            state.prototype_features.nbytes + state.prototype_actions.nbytes
        ),
        state_digest=digest,
        clock=lambda: 0.0,
    )
    controller = ServingPromotionController(
        actor,
        policy=declared,
        evaluator=evaluator,
        build_learner=build,
        state_digest=digest,
        max_pending=1,
    )
    new = GuardBatch(
        "synthetic-new",
        (("guard-new", "heldout"),),
        2,
        np.asarray([[0.3, -0.3, 0.4]], dtype=np.float64),
        np.asarray([[0.75, 0.0, 0.0, 1.0]], dtype=np.float64),
        "inner_guard",
    )
    old = GuardBatch(
        "synthetic-old",
        (("guard-old", "heldout"),),
        2,
        np.asarray([[10.3, -0.3, 0.4]]),
        np.asarray([[-0.5, 0.0, 0.0, -1.0]]),
        "inner_guard",
    )
    training_queries = {tuple(row) for row in source.snapshot_state().prototype_features}
    training_queries.update(tuple(row) for row in features)
    assert all(tuple(row) not in training_queries for guard in (new, old) for row in guard.features)
    return source, actor, runtime, clock, budget, targets, controller, new, old


def apply_delayed(fixture):
    _, _, runtime, clock, budget, targets, _, _, _ = fixture
    assert runtime.train_ready(max_updates=1) == () and budget.updates_completed == 0
    clock.advance_to(1)
    assert runtime.train_ready(max_updates=1) == () and budget.updates_completed == 0
    clock.advance_to(2)
    runtime.record_label(
        LabelArrival("label", "sample", "synthetic-train", 2, "serving-0", targets)
    )
    applied = runtime.train_ready(max_updates=1)
    assert len(applied) == budget.updates_completed == 1
    assert applied[0].observed_at == 0 and applied[0].arrived_at == applied[0].applied_at == 2


def prepare(fixture):
    _, _, runtime, _, _, _, controller, new, old = fixture
    return controller.prepare(
        runtime, new, old, now=2, training_ids=frozenset({("synthetic-train", "sample")})
    )


@pytest.fixture
def boundary(monkeypatch, tmp_path):
    creations = []
    current = {"environment": FakeEnvironment()}

    def factory():
        creations.append(current["environment"])
        return current["environment"]

    monkeypatch.setattr(port, "_make_environment", factory)
    monkeypatch.setattr(smoke, "version", lambda _: "fake-native-boundary")
    output = tmp_path / "trace.json"
    monkeypatch.setattr(sys, "argv", ["smoke", "--output", str(output)])
    return current, creations, output


def receipt(path):
    return json.loads(path.read_text(encoding="utf8"))


def fresh_output(boundary, monkeypatch, name):
    current, _, output = boundary
    current["environment"] = FakeEnvironment()
    path = output.with_name(name)
    monkeypatch.setattr(sys, "argv", ["smoke", "--output", str(path)])
    return current["environment"], path


def test_should_serve_actual_public_promotable_actor_with_owned_nested_bundle_and_nonempty_cache(
    boundary, monkeypatch
):
    current, creations, output = boundary
    metadata = {
        "nested": [None, True, 7, 0.5, "text", b"bytes", (1, 2)],
        "array": np.asarray([[1.0, 2.0]]),
    }
    supplied = actor_for(metadata=metadata)
    metadata["nested"][2] = 99
    supplied.serve(np.asarray([[0.2, -0.3, 0.4]]), now=1)
    before = supplied.serving_snapshot()
    assert before.cache and before.metadata["nested"][2] == 7
    backend = supplied._slot.bundle.learner
    calls, snapshots = [], []
    actual_predict, actual_snapshot = PromotableActor.predict, PromotableActor.serving_snapshot

    def predict(actor, features):
        calls.append((actor, features.copy()))
        return actual_predict(actor, features)

    def snapshot(actor):
        snapshots.append(actor)
        return actual_snapshot(actor)

    monkeypatch.setattr(PromotableActor, "predict", predict)
    monkeypatch.setattr(PromotableActor, "serving_snapshot", snapshot)
    monkeypatch.setattr(
        NumpyNearestContinuousBCLearner,
        "fork",
        lambda *_: pytest.fail("serving copied a replacement learner"),
    )
    monkeypatch.setattr(
        PromotableActor,
        "serve",
        lambda *_args, **_kwargs: pytest.fail("robot route mutated serving cache"),
    )
    assert smoke.main(actor=supplied) == 0
    result = receipt(output)
    assert creations == [current["environment"]] and current["environment"].close_calls == 1
    assert len(calls) == 2 and all(actor is supplied for actor, _ in calls)
    assert snapshots and all(actor is supplied for actor in snapshots)
    assert supplied._slot.bundle.learner is backend
    assert digest(before) == digest(supplied.serving_snapshot())
    for command in current["environment"].commands:
        np.testing.assert_array_equal(
            command[0], np.asarray([-0.5, 0.0, 0.0, -1.0], dtype=np.float32)
        )
    assert result["policy"] == "frozen_promotable_nearest_continuous_bc"
    assert result["state_guard_scope"] == "actual_promotable_actor_no_private_fork"
    assert result["actor_version"] == "serving-0"
    assert result["actor_generation"] == 0
    assert result["actor_generation_exact"] is result["actor_serving_bundle_exact"] is True
    for field in (
        "actor_configuration_exact",
        "actor_cache_exact",
        "actor_metadata_exact",
        "actor_last_cache_tick_exact",
    ):
        assert result[field] is True
    assert result["actor_version_exact"] is result["actor_backend_exact"] is True
    assert result["frozen_state_exact"] is result["original_owner_exact"] is True
    assert result["actor_prediction_attempts"] == result["actor_prediction_returns"] == 2
    assert result["actor_prediction_versions"] == ["serving-0"] * 2
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
    assert result["nearest_prediction_attempts"] == result["nearest_prediction_returns"] == 0
    assert result["nearest_selected_indices"] == []
    assert result["optimizer_updates"] == 0 and result["complete"] is True
    detached = supplied.serving_snapshot()
    detached.metadata["array"][:] = 8.0
    next(iter(detached.cache.values())).prediction[:] = 0.0
    assert digest(before) == digest(supplied.serving_snapshot())


def test_should_keep_candidate_update_private_then_admit_real_committed_controls(
    boundary, monkeypatch
):
    fixture = promotion_fixture()
    _, actor, runtime, _, budget, _, controller, _, _ = fixture
    before = actor.serving_snapshot()
    apply_delayed(fixture)
    assert runtime.candidate_snapshot().state.training_examples == 4
    assert digest(before) == digest(actor.serving_snapshot())
    assert smoke.main(actor=actor) == 0
    first = receipt(boundary[2])
    assert first["actor_version"] == "serving-0" and first["actor_generation"] == 0
    for command in boundary[0]["environment"].commands:
        assert command[0, 0] == -0.5
    ticket = prepare(fixture)
    assert ticket.report.decision.accepted is True
    committed = controller.commit(runtime, ticket)
    assert committed.generation == 1 and committed.actor_version == "candidate-0"
    environment, output = fresh_output(boundary, monkeypatch, "committed.json")
    assert smoke.main(actor=actor) == 0
    second = receipt(output)
    assert second["actor_version"] == "candidate-0" and second["actor_generation"] == 1
    assert second["actor_prediction_versions"] == ["candidate-0"] * 2
    assert second["actor_distance_candidates_completed"] == 8
    assert second["actor_distance_feature_values_completed"] == 24
    for command in environment.commands:
        np.testing.assert_array_equal(
            command[0], np.asarray([0.75, 0.0, 0.0, 1.0], dtype=np.float32)
        )
    assert budget.updates_completed == 1
    assert runtime.candidate_snapshot().state.completed_updates == 2


@pytest.mark.parametrize("when", ["prediction", "completed_step"])
def test_should_refuse_mid_rollout_real_commit_and_retain_completed_work(
    boundary, monkeypatch, when
):
    fixture = promotion_fixture()
    _, actor, runtime, _, _, _, controller, _, _ = fixture
    apply_delayed(fixture)
    ticket = prepare(fixture)
    environment = boundary[0]["environment"]
    receipts = []
    if when == "prediction":
        original = actor.predict

        def predict(features):
            value = original(features)
            receipts.append(controller.commit(runtime, ticket))
            return value

        monkeypatch.setattr(actor, "predict", predict)
    else:
        original = environment.step

        def step(command):
            value = original(command)
            receipts.append(controller.commit(runtime, ticket))
            return value

        monkeypatch.setattr(environment, "step", step)
    assert smoke.main(actor=actor) == 1
    result = receipt(boundary[2])
    completed = int(when == "completed_step")
    assert len(receipts) == 1 and actor.serving_snapshot().generation == 1
    assert len(environment.commands) == result["work"]["completed_steps"] == completed
    assert result["work"]["completed_control_physics_steps"] == completed * 5
    assert result["actor_prediction_attempts"] == result["actor_prediction_returns"] == 1
    assert result["complete"] is False and environment.close_calls == 1
    assert result["actor_generation_exact"] is result["actor_serving_bundle_exact"] is not True


def test_should_refuse_mid_rollout_real_rollback_then_allow_fresh_admission(boundary, monkeypatch):
    fixture = promotion_fixture()
    _, actor, runtime, _, _, _, controller, _, _ = fixture
    apply_delayed(fixture)
    committed = controller.commit(runtime, prepare(fixture))
    original = actor.predict

    def predict(features):
        result = original(features)
        controller.rollback(committed)
        return result

    monkeypatch.setattr(actor, "predict", predict)
    assert smoke.main(actor=actor) == 1
    result = receipt(boundary[2])
    assert not boundary[0]["environment"].commands
    assert result["actor_generation"] == 1 and result["actor_generation_exact"] is not True
    monkeypatch.setattr(actor, "predict", original)
    environment, output = fresh_output(boundary, monkeypatch, "rolled-back.json")
    assert smoke.main(actor=actor) == 0
    assert receipt(output)["actor_generation"] == 2
    assert receipt(output)["actor_version"] == "serving-0"
    assert environment.commands[0][0, 0] == -0.5


def test_should_recheck_actor_after_recipe_callback_immediately_before_control(
    boundary, monkeypatch
):
    fixture = promotion_fixture()
    _, actor, runtime, _, _, _, controller, _, _ = fixture
    apply_delayed(fixture)
    ticket = prepare(fixture)
    predicted, committed = [], []
    original_predict, original_recipe = actor.predict, smoke._check_visual_recipe

    def predict(features):
        result = original_predict(features)
        predicted.append(True)
        return result

    def recipe(current_port, result):
        original_recipe(current_port, result)
        if predicted and not committed:
            committed.append(controller.commit(runtime, ticket))

    monkeypatch.setattr(actor, "predict", predict)
    monkeypatch.setattr(smoke, "_check_visual_recipe", recipe)
    assert smoke.main(actor=actor) == 1
    result = receipt(boundary[2])
    assert len(committed) == 1 and not boundary[0]["environment"].commands
    assert result["validated_prediction_rows"] == 1
    assert result["actor_distance_candidates_completed"] == 2
    assert result["actor_serving_bundle_exact"] is None and result["complete"] is False


def mutate_bundle(actor, change):
    slot, bundle = actor._slot, actor._slot.bundle
    if change == "generation":
        actor._slot = replace(slot, generation=slot.generation + 1)
    elif change == "version":
        actor._slot = replace(slot, bundle=replace(bundle, version="unobserved"))
    elif change == "configuration":
        actor._slot = replace(
            slot, bundle=replace(bundle, configuration=ServingConfiguration(4, 2))
        )
    elif change == "metadata":
        actor._slot = replace(slot, bundle=replace(bundle, metadata={"route": [True]}))
    elif change == "cache":
        actor._slot = replace(
            slot, bundle=replace(bundle, cache={"a" * 64: CachedPrediction(3, np.zeros((1, 4)))})
        )
    elif change == "tick":
        actor._slot = replace(slot, bundle=replace(bundle, last_cache_tick=1))
    elif change == "model":
        state = bundle.learner.snapshot_state()
        state.prototype_actions[:] = 0.0
        bundle.learner.restore_state(state)
    elif change == "backend":
        replacement = NumpyNearestContinuousBCLearner(bundle.learner.policy)
        replacement.restore_state(bundle.learner.snapshot_state())
        actor._slot = replace(slot, bundle=replace(bundle, learner=replacement))
    else:
        raise AssertionError(change)


@pytest.mark.parametrize(
    "change",
    ["generation", "version", "configuration", "metadata", "cache", "tick", "model", "backend"],
)
def test_should_refuse_every_unobserved_complete_bundle_change_before_command(
    boundary, monkeypatch, change
):
    actor = actor_for(metadata={"route": [1]})
    original = actor.predict

    def predict(features):
        result = original(features)
        mutate_bundle(actor, change)
        return result

    monkeypatch.setattr(actor, "predict", predict)
    assert smoke.main(actor=actor) == 1
    result = receipt(boundary[2])
    assert not boundary[0]["environment"].commands
    assert result["actor_prediction_attempts"] == result["actor_prediction_returns"] == 1
    assert result["complete"] is False and result["actor_serving_bundle_exact"] is not True
    assert boundary[0]["environment"].close_calls == 1


def test_should_detect_cleanup_bundle_change_after_all_completed_controls(boundary, monkeypatch):
    actor = actor_for()
    environment = boundary[0]["environment"]
    original = environment.close

    def close():
        original()
        mutate_bundle(actor, "metadata")

    monkeypatch.setattr(environment, "close", close)
    assert smoke.main(actor=actor) == 1
    result = receipt(boundary[2])
    assert len(environment.commands) == result["work"]["completed_steps"] == 2
    assert result["work"]["completed_control_physics_steps"] == 10
    assert result["complete"] is False and result["actor_serving_bundle_exact"] is not True
    assert result["state_error"]


@pytest.mark.parametrize(
    "bad", ["object", "set", "nonfinite", "object_array", "cycle", "nodes", "bytes"]
)
def test_should_refuse_unsupported_or_unbounded_auxiliary_state_before_scene(boundary, bad):
    if bad == "object":
        value = object()
    elif bad == "set":
        value = {1, 2}
    elif bad == "nonfinite":
        value = float("nan")
    elif bad == "object_array":
        value = np.asarray([object()], dtype=object)
    elif bad == "cycle":
        value = []
        value.append(value)
    elif bad == "nodes":
        value = [None] * 4096
    else:
        value = b"x" * (1024**2 + 1)
    actor = actor_for(metadata={"bad": value})
    assert smoke.main(actor=actor) == 1
    result = receipt(boundary[2])
    assert not boundary[1] and result["creation_attempts"] == 0
    assert result["actor_prediction_attempts"] == 0 and result["complete"] is False


@pytest.mark.parametrize("bad", ["key", "entry", "expiry", "dtype", "shape", "nonfinite", "bounds"])
def test_should_validate_full_cache_entries_before_any_scene(boundary, bad):
    actor = actor_for()
    key, entry = "a" * 64, CachedPrediction(3, np.zeros((1, 4), dtype=np.float64))
    if bad == "key":
        key = "not-a-sha256"
    elif bad == "entry":
        entry = dict(expires_at=3, prediction=np.zeros((1, 4)))
    elif bad == "expiry":
        entry = CachedPrediction(True, entry.prediction)
    elif bad == "dtype":
        entry = CachedPrediction(3, entry.prediction.astype(np.float32))
    elif bad == "shape":
        entry = CachedPrediction(3, np.zeros((1, 3)))
    elif bad == "nonfinite":
        entry.prediction[0, 0] = np.inf
    else:
        entry.prediction[0, 0] = 1.01
    actor._slot = replace(actor._slot, bundle=replace(actor._slot.bundle, cache={key: entry}))
    assert smoke.main(actor=actor) == 1
    result = receipt(boundary[2])
    assert not boundary[1] and result["creation_attempts"] == 0


@pytest.mark.parametrize("field", ["generation", "last_cache_tick", "model_counter", "model_dtype"])
def test_should_refuse_equal_valued_wrong_types_in_late_public_observation(
    boundary, monkeypatch, field
):
    actor = actor_for()
    original_predict, original_snapshot = actor.predict, actor.serving_snapshot
    predicted = []

    def predict(features):
        result = original_predict(features)
        predicted.append(True)
        return result

    def snapshot():
        observed = original_snapshot()
        if not predicted:
            return observed
        if field == "generation":
            return replace(observed, generation=False)
        if field == "last_cache_tick":
            return replace(observed, last_cache_tick=False)
        if field == "model_counter":
            return replace(
                observed, model_state=replace(observed.model_state, completed_updates=True)
            )
        return replace(
            observed,
            model_state=replace(
                observed.model_state,
                prototype_actions=observed.model_state.prototype_actions.astype(np.float32),
            ),
        )

    monkeypatch.setattr(actor, "predict", predict)
    monkeypatch.setattr(actor, "serving_snapshot", snapshot)
    assert smoke.main(actor=actor) == 1
    result = receipt(boundary[2])
    assert not boundary[0]["environment"].commands
    assert result["actor_serving_bundle_exact"] is not True and result["complete"] is False


def test_should_preserve_unreturned_prediction_upper_bounds_without_completed_distance_work(
    boundary, monkeypatch
):
    actor = actor_for()

    def fail(_):
        raise RuntimeError("synthetic public prediction failure")

    monkeypatch.setattr(actor, "predict", fail)
    assert smoke.main(actor=actor) == 1
    result = receipt(boundary[2])
    assert result["actor_prediction_attempts"] == 1 and result["actor_prediction_returns"] == 0
    assert result["actor_distance_candidate_upper_bound"] == 2
    assert result["actor_distance_feature_upper_bound"] == 6
    assert (
        result["actor_distance_candidates_completed"]
        == result["actor_distance_feature_values_completed"]
        == 0
    )
    assert not boundary[0]["environment"].commands


@pytest.mark.parametrize("bad", ["version", "unversioned", "shape", "bounds", "nonfinite"])
def test_should_refuse_invalid_public_prediction_before_control_with_returned_work_retained(
    boundary, monkeypatch, bad
):
    actor = actor_for()
    original = actor.predict

    def predict(features):
        result = original(features)
        if bad == "version":
            return VersionedPrediction("unobserved", result.prediction)
        if bad == "unversioned":
            return result.prediction
        if bad == "shape":
            return VersionedPrediction(result.actor_version, result.prediction[:, :3])
        result.prediction[0, 0] = 1.1 if bad == "bounds" else np.nan
        return result

    monkeypatch.setattr(actor, "predict", predict)
    assert smoke.main(actor=actor) == 1
    result = receipt(boundary[2])
    assert result["actor_prediction_attempts"] == result["actor_prediction_returns"] == 1
    assert result["actor_distance_candidate_upper_bound"] == 2
    assert result["actor_distance_feature_upper_bound"] == 6
    assert (
        result["actor_distance_candidates_completed"]
        == result["actor_distance_feature_values_completed"]
        == 0
    )
    assert not boundary[0]["environment"].commands and result["complete"] is False


@pytest.mark.parametrize("role", ["train", "outer_selection", "final_test"])
def test_should_preserve_real_controller_guard_role_and_overlap_refusals(role):
    fixture = promotion_fixture()
    _, actor, runtime, _, _, _, controller, new, old = fixture
    apply_delayed(fixture)
    before = digest(actor.serving_snapshot())
    with pytest.raises(ValueError, match="inner_guard"):
        controller.prepare(runtime, replace(new, role=role), old, now=2, training_ids=frozenset())
    assert digest(actor.serving_snapshot()) == before


def test_should_refuse_training_overlap_in_synthetic_guard_fixture():
    fixture = promotion_fixture()
    _, actor, runtime, _, _, _, controller, new, old = fixture
    apply_delayed(fixture)
    before = digest(actor.serving_snapshot())
    with pytest.raises(ValueError, match="disjoint"):
        controller.prepare(runtime, new, old, now=2, training_ids=frozenset(new.sample_keys))
    assert digest(actor.serving_snapshot()) == before


@pytest.mark.parametrize("role", ["probe_train", "probe_test"])
def test_should_refuse_forged_probe_guard_roles_before_evaluation(role):
    fixture = promotion_fixture()
    _, actor, runtime, _, _, _, controller, new, old = fixture
    apply_delayed(fixture)
    object.__setattr__(new, "role", role)
    before = digest(actor.serving_snapshot())
    with pytest.raises(ValueError, match="inner_guard"):
        controller.prepare(runtime, new, old, now=2, training_ids=frozenset())
    assert digest(actor.serving_snapshot()) == before


@pytest.mark.parametrize("kind", ["cold", "probe", "clock", "both"])
def test_should_refuse_invalid_admission_before_scene(boundary, kind):
    actor = (
        actor_for(model(role="probe_train"))
        if kind == "probe"
        else actor_for(model(cold=kind == "cold"))
    )
    options = dict(actor=actor)
    if kind == "clock":
        options["observation_clock"] = EpisodeClock(49, True)
    elif kind == "both":
        options["learner"] = model()
    assert smoke.main(**options) == 1
    assert not boundary[1] and receipt(boundary[2])["creation_attempts"] == 0


def test_should_preserve_clock_encoding_on_actual_promotable_route(boundary):
    actor = actor_for(model(dimension=4))
    assert smoke.main(actor=actor, observation_clock=EpisodeClock(50, True)) == 0
    result = receipt(boundary[2])
    assert result["clock_configuration_exact"] is True
    assert result["clock_values"] == [0.0, 0.02]
    assert result["clock_encoding_attempts"] == result["clock_encoding_returns"] == 2
    assert result["actor_distance_feature_values_completed"] == 16
    assert result["episode"]["spec"]["observation_dim"] == 3


def test_should_refuse_changed_private_clock_before_control(boundary, monkeypatch):
    actor = actor_for(model(dimension=4))
    original = smoke.clocks.clock_observation

    def encode(observation, step, clock):
        result = original(observation, step, clock)
        object.__setattr__(clock, "elapsed", False)
        return result

    monkeypatch.setattr(smoke.clocks, "clock_observation", encode)
    assert smoke.main(actor=actor, observation_clock=EpisodeClock(50, True)) == 1
    result = receipt(boundary[2])
    assert not boundary[0]["environment"].commands
    assert result["clock_configuration_exact"] is False
    assert result["clock_encoding_attempts"] == result["clock_encoding_returns"] == 1
    assert result["actor_serving_bundle_exact"] is True


@pytest.mark.parametrize("kind", ["actor", "backend"])
def test_should_refuse_unlisted_actor_and_backend_subclasses_before_scene(boundary, kind):
    if kind == "actor":

        class DerivedActor(PromotableActor):
            pass

        actor = DerivedActor(
            model(),
            version="serving-0",
            configuration=ServingConfiguration(3, 2),
            feature_digest=digest,
            metadata={},
        )
    else:

        class DerivedBackend(NumpyNearestContinuousBCLearner):
            pass

        actor = actor_for()
        backend = DerivedBackend(actor._slot.bundle.learner.policy)
        backend.restore_state(actor._slot.bundle.learner.snapshot_state())
        actor._slot = replace(actor._slot, bundle=replace(actor._slot.bundle, learner=backend))
    assert smoke.main(actor=actor) == 1
    assert not boundary[1] and receipt(boundary[2])["creation_attempts"] == 0


def test_should_preserve_exclusive_output_before_actor_observations(boundary, monkeypatch):
    actor = actor_for()
    output = boundary[2]
    output.write_bytes(b"preserved prior result\n")
    monkeypatch.setattr(
        actor, "serving_snapshot", lambda: pytest.fail("existing receipt reached actor admission")
    )
    with pytest.raises(FileExistsError):
        smoke.main(actor=actor)
    assert output.read_bytes() == b"preserved prior result\n"
    assert not boundary[1]


@pytest.mark.parametrize("kind", ["slot", "bundle"])
def test_should_refuse_foreign_private_storage_before_public_callbacks(boundary, monkeypatch, kind):
    from types import SimpleNamespace

    actor = actor_for()
    if kind == "slot":
        actor._slot = SimpleNamespace(bundle=actor._slot.bundle, generation=0)
    else:
        actor._slot = replace(
            actor._slot, bundle=SimpleNamespace(learner=actor._slot.bundle.learner)
        )
    monkeypatch.setattr(
        actor, "serving_snapshot", lambda: pytest.fail("foreign storage reached public copy")
    )
    assert smoke.main(actor=actor) == 1
    assert not boundary[1]


@pytest.mark.parametrize("expiry", [0, 4])
def test_should_refuse_cache_outside_declared_tick_window_before_scene(boundary, expiry):
    actor = actor_for()
    cache = {"a" * 64: CachedPrediction(expiry, np.zeros((1, 4), dtype=np.float64))}
    actor._slot = replace(actor._slot, bundle=replace(actor._slot.bundle, cache=cache))
    assert smoke.main(actor=actor) == 1
    assert not boundary[1]


def test_should_bound_cache_aggregate_payload_before_finite_scanning(boundary):
    actor = actor_for()
    cache = {
        key * 64: CachedPrediction(3, np.zeros((20000, 4), dtype=np.float64)) for key in ("a", "b")
    }
    actor._slot = replace(actor._slot, bundle=replace(actor._slot.bundle, cache=cache))
    assert smoke.main(actor=actor) == 1
    assert "aggregate payload" in receipt(boundary[2])["error"]
    assert not boundary[1]


@pytest.mark.parametrize(
    "kind", ["wide_alias", "right_dictionary", "huge_text", "huge_key", "huge_integer"]
)
def test_should_refuse_auxiliary_expansion_and_payload_before_comparison(kind):
    from src.adapters.promotable_robot_binding import _owned_values_equal

    if kind == "wide_alias":
        child = [None] * 3000
        left = right = [child] * 3000
    elif kind == "right_dictionary":
        left, right = {}, {str(index): None for index in range(4097)}
    elif kind == "huge_text":
        left = right = "x" * (1024**2 + 1)
    elif kind == "huge_key":
        left = right = {"x" * (1024**2 + 1): None}
    else:
        left = right = 1 << (8 * 1024**2)
    with pytest.raises(ValueError, match="limit|payload"):
        _owned_values_equal(left, right)


@pytest.mark.parametrize("finite", [True, False])
def test_should_admit_only_finite_complex_numeric_metadata(boundary, finite):
    values = np.asarray([1 + 2j, 3j if finite else complex(float("inf"), 0)])
    actor = actor_for(metadata={"numeric": values})
    assert smoke.main(actor=actor) == (0 if finite else 1)
    assert bool(boundary[1]) is finite
