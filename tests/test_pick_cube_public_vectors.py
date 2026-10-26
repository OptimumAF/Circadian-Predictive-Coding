"""Owned public21 projection with fake native IO; no SGD or learned projection.

Only two actor checks append prototypes: one2-row append each, four rows total.
Every MLP is initialized/restored from existing deterministic fixtures.
"""

from dataclasses import FrozenInstanceError, asdict, replace
import json
import sys

import numpy as np
import pytest

from scripts import maniskill_continuous_smoke as smoke
from src.adapters.numpy_continuous_bc import NumpyContinuousBCLearner
from src.adapters.numpy_nearest_continuous_bc import (
    NearestContinuousBCPolicy,
    NumpyNearestContinuousBCLearner,
)
from src.app.actor_shadow import StableActor
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)
from src.core.continuous_episode_clock import EpisodeClock
from src.core.observation_standardization import ObservationStandardization
from src.core.pick_cube_camera_context import camera_context_recipe, pick_cube_camera_context
from src.core.pick_cube_rgb_features import pick_cube_rgb_recipe
from src.core import pick_cube_public_vectors as public
from src.infra import maniskill_continuous as port
from test_maniskill_bc_smoke import assert_same_state, learner
from test_maniskill_rgb_observations import RGBEnvironment, packet, public_feature_oracle


def observation(index=0):
    return tuple(float(index + coordinate) for coordinate in range(21)) + (0.25,) * 768


def raw_episode(seed=17, *, camera=False, role="train"):
    context = (
        pick_cube_camera_context().context
        if camera
        else "pick-cube"
        if seed == 17
        else "pick-cube-train-19"
    )
    identity = f"{context}-teacher" if camera else f"pick-cube-task-teacher-{seed}"
    spec = ContinuousActionSpec(789, port.ACTION_UNITS, (-1.0,) * 4, (1.0,) * 4)
    return ContinuousEpisode(
        identity,
        context,
        role,
        spec,
        tuple(
            ContinuousTransition(
                index,
                observation(index),
                (0.125, -0.5, 0.75, 1.0),
                float(index == 1),
                observation(index + 1),
                False,
                index == 1,
            )
            for index in range(2)
        ),
    )


def small_model(dimension=21):
    return learner(spec=ContinuousActionSpec(dimension, port.ACTION_UNITS, (-1.0,) * 4, (1.0,) * 4))


@pytest.fixture
def boundary(monkeypatch, tmp_path):
    environment, calls = RGBEnvironment(), []

    def factory(**options):
        calls.append(options)
        return environment

    monkeypatch.setattr(port, "_make_environment", factory)
    monkeypatch.setattr(smoke, "version", lambda _: "fake-public-vector-boundary")
    return environment, calls, tmp_path


def test_should_freeze_only_fixed_public_projection_and_own_validated_metadata():
    projection = public.public_vector_projection()
    assert type(projection) is public.PickCubePublicVectorProjection
    assert asdict(projection) == dict(
        definition="pick_cube_public_qpos9_qvel9_goal3_v1",
        source_dim=789,
        feature_dim=21,
        feature_order=("qpos9", "qvel9", "public_goal3"),
    )
    owned = public.validate_public_vector_projection(projection)
    assert owned == projection and owned is not projection
    with pytest.raises(FrozenInstanceError):
        projection.feature_dim = 20


@pytest.mark.parametrize(
    "damage",
    [
        "none",
        "mapping",
        "subclass",
        "source_type",
        "feature_type",
        "dimension",
        "definition",
        "feature_order",
        "order_type",
    ],
)
def test_should_refuse_arbitrary_or_untyped_projection_records(damage):
    projection = public.public_vector_projection()
    if damage == "none":
        projection = None
    elif damage == "mapping":
        projection = asdict(projection)
    elif damage == "subclass":

        class Derived(public.PickCubePublicVectorProjection):
            pass

        projection = Derived()
    else:
        field, value = {
            "source_type": ("source_dim", 789.0),
            "feature_type": ("feature_dim", 21.0),
            "dimension": ("feature_dim", 20),
            "definition": ("definition", "selected_subset"),
            "feature_order": ("feature_order", ("public_goal3", "qvel9", "qpos9")),
            "order_type": ("feature_order", ["qpos9", "qvel9", "public_goal3"]),
        }[damage]
        object.__setattr__(projection, field, value)
    with pytest.raises(ValueError):
        public.validate_public_vector_projection(projection)


def test_should_project_exact_public_order_and_preserve_arbitrary_action_spec():
    projection = public.public_vector_projection()
    source = ContinuousActionSpec(789, ("m", "m", "constant"), (-2.0, 3.0, 1.0), (6.0, 5.0, 1.0))
    projected = public.public_vector_spec(source, projection)
    assert projected.observation_dim == 21
    assert projected.action_units == source.action_units
    assert (
        projected.lower_bounds == source.lower_bounds
        and projected.upper_bounds == source.upper_bounds
    )
    assert projected is not source
    before = np.random.get_state()
    result = public.public_vector_observation(
        observation(), projection, source_recipe=pick_cube_rgb_recipe()
    )
    after = np.random.get_state()
    assert result == tuple(float(index) for index in range(21))
    assert type(result) is tuple and all(type(value) is float for value in result)
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])


@pytest.mark.parametrize(
    "damage",
    [
        "list",
        "short",
        "long",
        "nan_prefix",
        "infinite_dropped",
        "non_native_prefix",
        "boolean_dropped",
        "rgb_low",
        "rgb_high",
    ],
)
def test_should_validate_every_source_value_before_discarding_rgb(damage):
    values = list(observation())
    if damage == "list":
        values = list(values)
    elif damage == "short":
        values = tuple(values[:-1])
    elif damage == "long":
        values = tuple(values + [0.25])
    else:
        index, value = {
            "nan_prefix": (0, float("nan")),
            "infinite_dropped": (788, float("inf")),
            "non_native_prefix": (0, np.float64(0.0)),
            "boolean_dropped": (788, False),
            "rgb_low": (21, -1e-12),
            "rgb_high": (788, 1.0 + 1e-12),
        }[damage]
        values[index] = value
        values = tuple(values)
    with pytest.raises(ValueError):
        public.public_vector_observation(
            values, public.public_vector_projection(), source_recipe=pick_cube_rgb_recipe()
        )


@pytest.mark.parametrize("pixel", [0.0, 1.0])
def test_should_accept_exact_rgb_endpoints_without_clipping_public_vectors(pixel):
    values = (-1e100,) + observation()[1:21] + (pixel,) * 768
    result = public.public_vector_observation(
        values, public.public_vector_projection(), source_recipe=pick_cube_rgb_recipe()
    )
    assert result == values[:21] and result[0] == -1e100


@pytest.mark.parametrize("camera", [False, True])
def test_should_require_exact_declared_recipe_and_camera_binding(camera):
    context = pick_cube_camera_context() if camera else None
    recipe = camera_context_recipe(context) if camera else pick_cube_rgb_recipe()
    projection = public.public_vector_projection()
    assert (
        public.public_vector_observation(
            observation(), projection, source_recipe=recipe, camera_context=context
        )
        == observation()[:21]
    )
    wrong = replace(recipe, pixel_divisor=255)
    with pytest.raises(ValueError):
        public.public_vector_observation(
            observation(), projection, source_recipe=wrong, camera_context=context
        )
    if camera:
        with pytest.raises(ValueError):
            public.public_vector_observation(observation(), projection, source_recipe=recipe)


@pytest.mark.parametrize("recipe", [None, asdict(pick_cube_rgb_recipe())])
def test_should_refuse_missing_or_untyped_source_recipe(recipe):
    with pytest.raises(ValueError):
        public.public_vector_observation(
            observation(), public.public_vector_projection(), source_recipe=recipe
        )


@pytest.mark.parametrize("seed,camera", [(17, False), (19, False), (17, True)])
def test_should_project_owned_complete_train_episode_without_relabeling_controls(seed, camera):
    source = raw_episode(seed, camera=camera)
    original = asdict(source)
    context = pick_cube_camera_context() if camera else None
    recipe = camera_context_recipe(context) if camera else pick_cube_rgb_recipe()
    result = public.public_vector_training_episode(
        source, public.public_vector_projection(), source_recipe=recipe, camera_context=context
    )
    assert (
        result.episode_id == source.episode_id
        and result.context == source.context
        and result.role == "train"
    )
    assert (
        result.spec.observation_dim == 21 and result.spec.action_units == source.spec.action_units
    )
    for before, after in zip(source.transitions, result.transitions, strict=True):
        assert (
            after.observation == before.observation[:21]
            and after.next_observation == before.next_observation[:21]
        )
        assert (after.step, after.action, after.reward, after.terminated, after.truncated) == (
            before.step,
            before.action,
            before.reward,
            before.terminated,
            before.truncated,
        )
        assert after is not before
    assert asdict(source) == original
    object.__setattr__(source.transitions[0], "action", (0.0,) * 4)
    assert result.transitions[0].action == (0.125, -0.5, 0.75, 1.0)


@pytest.mark.parametrize("role", ["inner_guard", "probe_train", "probe_test", "final_test"])
def test_should_refuse_non_train_episode_before_projection(monkeypatch, role):
    source = raw_episode(role=role)
    monkeypatch.setattr(
        public,
        "public_vector_observation",
        lambda *_a, **_k: pytest.fail("heldout observation projected"),
    )
    with pytest.raises(ValueError):
        public.public_vector_training_episode(
            source, public.public_vector_projection(), source_recipe=pick_cube_rgb_recipe()
        )


@pytest.mark.parametrize(
    "damage", ["unknown_context", "lost_camera", "terminal_rgb", "source_dimension", "too_long"]
)
def test_should_refuse_unbound_or_corrupt_complete_episode(damage):
    source = raw_episode(camera=damage == "lost_camera")
    if damage == "unknown_context":
        source = replace(source, context="pick-cube-probe-23")
    elif damage == "terminal_rgb":
        last = source.transitions[-1]
        values = last.next_observation[:-1] + (1.01,)
        source = replace(
            source, transitions=source.transitions[:-1] + (replace(last, next_observation=values),)
        )
    elif damage == "source_dimension":
        source = replace(
            source,
            spec=replace(source.spec, observation_dim=788),
            transitions=tuple(
                replace(
                    row,
                    observation=row.observation[:-1],
                    next_observation=row.next_observation[:-1],
                )
                for row in source.transitions
            ),
        )
    elif damage == "too_long":
        source = replace(
            source,
            transitions=tuple(
                ContinuousTransition(
                    index,
                    observation(index),
                    (0.0,) * 4,
                    0.0,
                    observation(index + 1),
                    False,
                    index == 50,
                )
                for index in range(51)
            ),
        )
    with pytest.raises(ValueError):
        public.public_vector_training_episode(
            source, public.public_vector_projection(), source_recipe=pick_cube_rgb_recipe()
        )


@pytest.mark.parametrize("camera", [False, True])
def test_should_feed_exact21_to_frozen_mlp_while_exporting_unchanged789_trace(
    boundary, monkeypatch, camera
):
    environment, calls, directory = boundary
    owner = small_model()
    before = owner.snapshot_state()
    predict, received = NumpyContinuousBCLearner.predict, []

    def observed(child, values):
        received.append(values.copy())
        return predict(child, values)

    monkeypatch.setattr(NumpyContinuousBCLearner, "predict", observed)
    monkeypatch.setattr(
        NumpyContinuousBCLearner,
        "train_batch",
        lambda *_a, **_k: pytest.fail("projection trained MLP"),
    )
    extra = dict(camera_context=pick_cube_camera_context()) if camera else {}
    result = smoke.run_frozen_rollout(
        directory / "projected.json",
        learner=owner,
        visual_observation=True,
        observation_projection=public.public_vector_projection(),
        **extra,
    ).report
    assert result["complete"] is True and result["projection_exact"] is True
    assert result["observation_projection"] == asdict(public.public_vector_projection())
    assert result["projection_work"] == dict(
        observation_attempts=2,
        observation_returns=2,
        source_values_upper_bound=1578,
        source_values_validated=1578,
        projected_values=42,
        projected_numeric_bytes=336,
    )
    assert result["episode"]["spec"]["observation_dim"] == 789
    assert result["visual_work"]["observation_returns"] == 3
    assert result["visual_work"]["returned_feature_values"] == 3 * 789
    assert result["prediction_attempts"] == result["prediction_returns"] == 2
    assert result["frozen_state_exact"] is result["original_owner_exact"] is True
    assert result["optimizer_updates"] == 0 and environment.close_calls == 1
    assert calls == [dict(visual_observation=True, **extra)]
    assert_same_state(before, owner.snapshot_state())
    expected = np.tanh(np.array([[0.2, -0.3, 0.4, -0.5]])).astype(np.float32)
    assert len(received) == 2
    for values, row, command in zip(
        received, result["episode"]["transitions"], environment.commands, strict=True
    ):
        assert values.shape == (1, 21) and values.dtype == np.float64
        np.testing.assert_array_equal(values[0], row["observation"][:21])
        assert len(row["observation"]) == len(row["next_observation"]) == 789
        np.testing.assert_array_equal(command, expected)
        assert row["action"] == tuple(float(value) for value in command[0])


@pytest.mark.parametrize("camera", [False, True])
def test_should_project_for_actual_nearest_serving_actor_without_fork_or_state_change(
    boundary, monkeypatch, camera
):
    """ONE2-row nearest append per case; no SGD and no per-rollout fork."""
    environment, _, directory = boundary
    spec = ContinuousActionSpec(21, port.ACTION_UNITS, (-1.0,) * 4, (1.0,) * 4)
    stats = ObservationStandardization((0.0,) * 21, (1.0,) * 21, 2)
    owner = NumpyNearestContinuousBCLearner(NearestContinuousBCPolicy(spec, stats))
    rows = [public_feature_oracle(packet(index))[:21] for index in (0, 1)]
    actions = np.array([[0.125, -1.0, 1.0, 0.0], [-0.5, 0.5, 0.0, -1.0]], dtype=np.float64)
    owner.train_batch(np.asarray(rows, dtype=np.float64), actions)
    actor = StableActor(owner, version="public-serving")
    initial = actor.snapshot_state()
    predict, served = actor.predict, []

    def observed(values):
        served.append(values.copy())
        return predict(values)

    monkeypatch.setattr(actor, "predict", observed)
    monkeypatch.setattr(
        NumpyNearestContinuousBCLearner, "fork", lambda *_: pytest.fail("actual actor was forked")
    )
    extra = dict(camera_context=pick_cube_camera_context()) if camera else {}
    result = smoke.run_frozen_rollout(
        directory / "actor.json",
        actor=actor,
        visual_observation=True,
        observation_projection=public.public_vector_projection(),
        **extra,
    ).report
    assert result["complete"] is True and result["actor_version"] == "public-serving"
    assert result["state_guard_scope"] == "actual_serving_actor_no_private_fork"
    assert result["actor_prediction_attempts"] == result["actor_prediction_returns"] == 2
    assert result["actor_distance_candidates_completed"] == 4
    assert result["actor_distance_feature_values_completed"] == 84
    assert result["projection_work"]["projected_values"] == 42
    assert result["nearest_prediction_returns"] == 0 and result["nearest_selected_indices"] == []
    assert len(served) == 2 and all(values.shape == (1, 21) for values in served)
    after = actor.snapshot_state()
    assert after.actor_version == initial.actor_version
    assert after.state.policy == initial.state.policy
    assert after.state.completed_updates == initial.state.completed_updates == 1
    np.testing.assert_array_equal(after.state.prototype_features, initial.state.prototype_features)
    np.testing.assert_array_equal(after.state.prototype_actions, initial.state.prototype_actions)
    for command, action in zip(environment.commands, actions, strict=True):
        np.testing.assert_array_equal(command[0], action.astype(np.float32))


@pytest.mark.parametrize(
    "damage",
    ["zero_policy", "nonvisual", "dimension", "clock", "boolean", "mapping"],
)
def test_should_refuse_incompatible_projection_before_model_fork_or_native(
    boundary, monkeypatch, damage
):
    environment, calls, directory = boundary
    owner = small_model(789 if damage == "dimension" else 21)
    arguments = dict(
        learner=owner,
        visual_observation=True,
        observation_projection=public.public_vector_projection(),
    )
    if damage == "zero_policy":
        arguments.pop("learner")
    elif damage == "nonvisual":
        arguments["visual_observation"] = False
    elif damage == "clock":
        arguments["observation_clock"] = EpisodeClock(50, True)
    elif damage == "boolean":
        arguments["observation_projection"] = True
    elif damage == "mapping":
        arguments["observation_projection"] = asdict(public.public_vector_projection())
    monkeypatch.setattr(owner, "fork", lambda: pytest.fail("invalid projection forked"))
    monkeypatch.setattr(
        smoke, "version", lambda _: pytest.fail("invalid projection queried runtime")
    )
    result = smoke.run_frozen_rollout(directory / "refused.json", **arguments).report
    assert result["complete"] is False and result["creation_attempts"] == 0
    assert calls == [] and environment.reset_calls == []


def test_should_refuse_corrupt_dropped_rgb_before_any_prediction_or_control(boundary, monkeypatch):
    environment, _, directory = boundary
    observed = port.ManiSkillPickCubeEpisode.observation

    def corrupted(session):
        values = observed.fget(session)
        return values[:-1] + (1.01,)

    monkeypatch.setattr(port.ManiSkillPickCubeEpisode, "observation", property(corrupted))
    result = smoke.run_frozen_rollout(
        directory / "corrupt.json",
        learner=small_model(),
        visual_observation=True,
        observation_projection=public.public_vector_projection(),
    ).report
    assert result["complete"] is False and "RGB" in result["error"]
    assert result["projection_work"] == dict(
        observation_attempts=1,
        observation_returns=0,
        source_values_upper_bound=789,
        source_values_validated=0,
        projected_values=0,
        projected_numeric_bytes=0,
    )
    assert result["prediction_attempts"] == 0 and environment.commands == []
    assert result["visual_work"]["observation_returns"] == 1
    assert environment.close_calls == 1


@pytest.mark.parametrize("phase", ["after_step", "cleanup"])
def test_should_refuse_projection_metadata_swap_and_keep_paid_controls(
    boundary, monkeypatch, phase
):
    environment, _, directory = boundary
    check = smoke._check_projection

    def changed(projection, result):
        ready = (
            bool(environment.commands) if phase == "after_step" else bool(environment.close_calls)
        )
        if projection is not None and ready:
            result["observation_projection"]["feature_dim"] = 21.0
        return check(projection, result)

    monkeypatch.setattr(smoke, "_check_projection", changed)
    owner = small_model()
    initial = owner.snapshot_state()
    result = smoke.run_frozen_rollout(
        directory / "drift.json",
        learner=owner,
        visual_observation=True,
        observation_projection=public.public_vector_projection(),
    ).report
    assert result["complete"] is False and result["projection_exact"] is None
    assert result["projection_error"]
    assert (
        result["work"]["completed_steps"]
        == len(environment.commands)
        == (1 if phase == "after_step" else 2)
    )
    assert result["projection_work"]["observation_returns"] == len(environment.commands)
    assert environment.close_calls == 1
    assert_same_state(initial, owner.snapshot_state())


@pytest.mark.parametrize("phase", ["before_control", "cleanup"])
def test_should_refuse_owner_mutation_in_projection_metadata_callback(boundary, monkeypatch, phase):
    environment, _, directory = boundary
    owner = small_model()
    check, mutations = smoke._check_projection, []

    def changed(projection, result):
        ready = (
            result["prediction_returns"] == 1
            if phase == "before_control"
            else bool(environment.close_calls)
        )
        if projection is not None and ready and not mutations:
            state = owner.snapshot_state()
            state.velocity[0].flat[0] += 0.5
            owner.restore_state(state)
            mutations.append(True)
        return check(projection, result)

    monkeypatch.setattr(smoke, "_check_projection", changed)
    result = smoke.run_frozen_rollout(
        directory / "mutated.json",
        learner=owner,
        visual_observation=True,
        observation_projection=public.public_vector_projection(),
    ).report
    assert result["complete"] is False and mutations == [True]
    assert result["original_owner_exact"] is False
    assert (
        len(environment.commands)
        == result["work"]["completed_steps"]
        == (0 if phase == "before_control" else 2)
    )
    assert result["prediction_returns"] == (1 if phase == "before_control" else 2)
    assert result["projection_work"]["observation_returns"] == result["prediction_returns"]
    assert environment.close_calls == 1


def test_should_preserve_unprojected_default_keys_and_raw789_input(boundary, monkeypatch):
    _, calls, directory = boundary
    monkeypatch.setattr(
        smoke.public_vectors,
        "public_vector_observation",
        lambda *_a, **_k: pytest.fail("default projected"),
    )
    result = smoke.run_frozen_rollout(
        directory / "default.json", learner=small_model(789), visual_observation=True
    ).report
    assert result["complete"] is True
    assert not {"observation_projection", "projection_exact", "projection_work"}.intersection(
        result
    )
    assert result["episode"]["spec"]["observation_dim"] == 789
    assert calls == [dict(visual_observation=True)]


@pytest.mark.parametrize("damage", ["reset_zero", "inflate_one", "boolean"])
def test_should_refuse_consistent_but_false_projection_work_at_cleanup(
    boundary, monkeypatch, damage
):
    environment, _, directory = boundary
    check = smoke._check_projection

    def changed(projection, result):
        if projection is not None and environment.close_calls:
            rows = 0 if damage == "reset_zero" else result["prediction_attempts"] + 1
            result["projection_work"] = dict(
                observation_attempts=rows,
                observation_returns=rows,
                source_values_upper_bound=rows * 789,
                source_values_validated=rows * 789,
                projected_values=rows * 21,
                projected_numeric_bytes=rows * 168,
            )
            if damage == "boolean":
                result["projection_work"]["observation_returns"] = True
        return check(projection, result)

    monkeypatch.setattr(smoke, "_check_projection", changed)
    report = smoke.run_frozen_rollout(
        directory / "false-work.json",
        learner=small_model(),
        visual_observation=True,
        observation_projection=public.public_vector_projection(),
    ).report
    assert report["complete"] is False and report["projection_exact"] is None
    assert "work changed" in report["projection_error"]
    assert report["prediction_attempts"] == report["work"]["completed_steps"] == 2
    assert len(environment.commands) == 2 and environment.close_calls == 1


def test_should_opt_in_through_existing_cli_and_preserve_exclusive_output(boundary, monkeypatch):
    environment, _, directory = boundary
    output = directory / "cli.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["smoke", "--output", str(output), "--visual-observation", "--public-vector-only"],
    )
    assert smoke.main(learner=small_model()) == 0
    report = json.loads(output.read_text(encoding="utf8"))
    assert report["complete"] is True and report["projection_exact"] is True
    assert report["observation_projection"]["feature_dim"] == 21
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        smoke.main(learner=small_model())
    assert output.read_bytes() == before and len(environment.commands) == 2
