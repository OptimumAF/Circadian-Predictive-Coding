"""Literal frozen owners and fake native IO; no recordings, training or SDK."""

from dataclasses import replace
import json
from typing import cast

import numpy as np
import pytest

from scripts import maniskill_continuous_smoke as smoke
from src.adapters import numpy_nearest_continuous_bc as near
from src.adapters import numpy_continuous_bc as bp
from src.adapters import numpy_continuous_pc as pc
from src.app.action_history_encoding import ActionHistoryEncoding
from src.core.cube_taskspace_state import cube_taskspace_spec
from src.core.coupled_circadian_predictive_coding import CoupledCircadianConfig
from src.core.observation_standardization import ObservationStandardization
from src.core.pick_cube_taskspace_history import PickCubeTaskSpaceHistoryRecipe
from src.core.pick_cube_action_history import PickCubeActionHistoryRecipe
from src.infra import maniskill_continuous as native
from test_state_quaternion_robot_learning import QuaternionEnvironment

WORK = dict(
    literal_sources=0,
    creations=0,
    nearest_predictions=0,
    query_rows=0,
    candidates=0,
    features=0,
    neural_predictions=0,
    neural_rows=0,
    fake_controls=0,
    snapshots=0,
    snapshot_payload=0,
    forbidden=0,
)
CAPS = dict(
    literal_sources=144,
    creations=128,
    nearest_predictions=256,
    query_rows=256,
    candidates=512,
    features=6144,
    neural_predictions=16,
    neural_rows=16,
    fake_controls=256,
    snapshots=65536,
    snapshot_payload=256 * 1024**2,
    forbidden=0,
)


def charge(key, amount=1):
    WORK[key] += amount
    assert WORK[key] <= CAPS[key], "permanent fixture cap: " + key


def forbidden(*args, **kwargs):
    charge("forbidden")


@pytest.fixture(autouse=True)
def count_work(monkeypatch):
    for kind in (
        near.NumpyNearestContinuousBCLearner,
        bp.NumpyContinuousBCLearner,
        pc.NumpyContinuousPCLearner,
    ):
        monkeypatch.setattr(kind, "__init__", forbidden)
        monkeypatch.setattr(kind, "train_batch", forbidden)
        original_snapshot = kind.snapshot_state

        def counted(owner, original=original_snapshot):
            charge("snapshots")
            state = original(owner)
            arrays = (
                (state.prototype_features, state.prototype_actions)
                if type(state) is near.NearestContinuousBCSnapshot
                else state.parameters + state.velocity
            )
            charge("snapshot_payload", sum(array.nbytes for array in arrays))
            return state

        monkeypatch.setattr(kind, "snapshot_state", counted)
    prediction = near.NumpyNearestContinuousBCLearner.predict_with_work

    def nearest(owner, values):
        charge("nearest_predictions")
        charge("query_rows", len(values))
        candidates = len(values) * owner._state.training_examples
        charge("candidates", candidates)
        charge("features", candidates * owner._policy.spec.observation_dim)
        return prediction(owner, values)

    monkeypatch.setattr(near.NumpyNearestContinuousBCLearner, "predict_with_work", nearest)
    for kind in (bp.NumpyContinuousBCLearner, pc.NumpyContinuousPCLearner):
        original_predict = kind.predict

        def predicted(owner, values, original=original_predict):
            charge("neural_predictions")
            charge("neural_rows", len(values))
            return original(owner, values)

        monkeypatch.setattr(kind, "predict", predicted)
    original_step = native.ManiSkillPickCubeEpisode.step

    def stepped(port, action):
        charge("fake_controls")
        return original_step(port, action)

    monkeypatch.setattr(native.ManiSkillPickCubeEpisode, "step", stepped)
    monkeypatch.setattr(native, "_make_environment", forbidden)
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    monkeypatch.setattr(bp, "fit_training_standardization", forbidden)
    monkeypatch.setattr(smoke, "PickCubeCorrectionQuery", forbidden)


def source(*, rows=2, fitted=False, role="train", dimension=12):
    charge("literal_sources")
    spec = replace(cube_taskspace_spec(), observation_dim=dimension)
    policy: near.NearestContinuousBCPolicy = near.FixedInputNearestContinuousBCPolicy(
        spec, training_role=role
    )
    if fitted:
        policy = near.NearestContinuousBCPolicy(
            spec, ObservationStandardization((0.0,) * dimension, (1.0,) * dimension, 1)
        )
    owner = object.__new__(near.NumpyNearestContinuousBCLearner)
    owner._policy = policy
    owner._state = near._snapshot(
        near.NearestContinuousBCSnapshot(
            policy,
            np.zeros((rows, dimension)),
            np.tile((0.1, -0.2, 0.3, -0.4), (rows, 1)),
            1 if rows else 0,
            rows,
        ),
        policy,
    )
    return owner


@pytest.fixture
def boundary(monkeypatch, tmp_path):
    environment = QuaternionEnvironment()

    def sparse():
        environment.bad_reward = np.array([float(len(environment.commands) == 2)])

    environment.after_step = sparse
    created = []

    def factory():
        charge("creations")
        created.append(True)
        return environment

    monkeypatch.setattr(native, "_make_environment", factory)
    monkeypatch.setattr(smoke, "version", lambda _: "fake-native-probe")
    return environment, created, tmp_path / "fresh.json"


def run(learner, boundary, **options):
    return smoke.run_frozen_rollout(
        boundary[2],
        learner=learner,
        action_history_recipe=PickCubeTaskSpaceHistoryRecipe(),
        **options,
    ).report


def test_should_score_only_original24_with_frozen_source_and_native32_history(boundary):
    learner = source()
    identity = smoke._pin_nearest_probe(learner)
    report = run(learner, boundary, probe_test_seed=24)
    assert report["complete"], report
    assert report["role"] == report["episode"]["role"] == "probe_test"
    assert report["seed"] == report["probe_test_seed"] == 24 and report["scoring_only"] is True
    assert report["episode"]["episode_id"] == "pick-cube-probe-test-smoke-24"
    assert report["prediction_attempts"] == report["prediction_returns"] == 2
    assert report["nearest_distance_candidates_completed"] == 4
    assert (
        report["frozen_state_exact"]
        is report["original_owner_exact"]
        is report["action_history_exact"]
        is True
    )
    assert report["optimizer_updates"] == 0 and "corrections" not in report
    assert report["work"]["completed_steps"] == 2
    assert report["nearest_control_calls"] == dict(attempted=2, returned=2)
    assert boundary[0].reset_calls == [24] and boundary[0].close_calls == 1
    assert all(value.dtype == np.dtype("float32") for value in boundary[0].commands)
    assert report["episode"]["transitions"][0]["action"] == tuple(
        float(v) for v in boundary[0].commands[0][0]
    )
    smoke._require_nearest_probe(identity)
    assert json.loads(boundary[2].read_text())["complete"]


@pytest.mark.parametrize("seed", [23, 29, 17, True, 24.0, "24"])
def test_should_refuse_wrong_reserved_seed_before_any_model_observation(
    boundary, monkeypatch, seed
):
    learner = source()
    monkeypatch.setattr(learner, "snapshot_state", lambda: pytest.fail("public model observed"))
    result = run(learner, boundary, probe_test_seed=seed)
    assert not result["complete"] and result["creation_attempts"] == 0
    assert boundary[1] == []


@pytest.mark.parametrize("seed", [31, 37])
def test_should_keep_nearest_guard_scoring_refused(boundary, seed):
    result = run(source(), boundary, guard_seed=seed)
    assert not result["complete"] and result["creation_attempts"] == 0 and not boundary[1]


@pytest.mark.parametrize("mode", ["fitted", "empty", "probe_role", "wrong_dimension"])
def test_should_refuse_nonmatching_source_before_creation(boundary, mode):
    learner = source(
        fitted=mode == "fitted",
        rows=0 if mode == "empty" else 2,
        role="probe_train" if mode == "probe_role" else "train",
        dimension=47 if mode == "wrong_dimension" else 12,
    )
    result = run(learner, boundary, probe_test_seed=24)
    assert not result["complete"] and not boundary[1]


@pytest.mark.parametrize(
    "option",
    [
        "actor",
        "observation_clock",
        "visual_observation",
        "corrective_queries",
        "readiness_corrections",
        "observed_grasp_corrections",
        "goal_consistent_corrections",
        "monitor_joint_positions",
        "observation_projection",
        "mass_context",
        "lighting_context",
        "camera_context",
        "missing_recipe",
        "zero_recipe",
        "history47",
        "train_seed",
    ],
)
def test_should_refuse_optional_modes_before_creation(boundary, option):
    args = dict(
        learner=source(), probe_test_seed=24, action_history_recipe=PickCubeTaskSpaceHistoryRecipe()
    )
    if option == "missing_recipe":
        args["action_history_recipe"] = None
    elif option == "zero_recipe":
        args["action_history_recipe"] = PickCubeTaskSpaceHistoryRecipe(mode="zero")
    elif option == "history47":
        args["action_history_recipe"] = PickCubeActionHistoryRecipe()
    elif option == "train_seed":
        args[option] = 19
    else:
        args[option] = (
            True
            if option
            in (
                "visual_observation",
                "corrective_queries",
                "readiness_corrections",
                "observed_grasp_corrections",
                "goal_consistent_corrections",
                "monitor_joint_positions",
            )
            else object()
        )
    result = smoke.run_frozen_rollout(boundary[2], **args).report
    assert not result["complete"] and not boundary[1]


def damage(owner, field):
    if field == "state":
        owner._state = owner.snapshot_state()
    elif field == "owner_policy":
        owner._policy = replace(owner._policy)
    elif field == "state_policy":
        object.__setattr__(owner._state, "policy", replace(owner._state.policy))
    elif field in ("owner_spec", "state_spec"):
        policy = owner._policy if field == "owner_spec" else owner._state.policy
        object.__setattr__(policy, "spec", replace(policy.spec))
    elif field in ("features", "actions"):
        name = "prototype_" + field
        object.__setattr__(owner._state, name, getattr(owner._state, name).copy())
    elif field == "counter":
        object.__setattr__(owner._state, "completed_updates", 2)
    elif field == "units":
        object.__setattr__(
            owner._state.policy.spec,
            "action_units",
            tuple(list(owner._state.policy.spec.action_units)),
        )
    else:
        values = owner._state.prototype_features
        values.setflags(write=True)
        values[0, 0] += 1.0


@pytest.mark.parametrize(
    "field",
    [
        "state",
        "owner_policy",
        "state_policy",
        "owner_spec",
        "state_spec",
        "features",
        "actions",
        "counter",
        "units",
        "values",
    ],
)
def test_should_refuse_changed_or_equal_replaced_native_binding_before_creation(
    boundary, monkeypatch, field
):
    learner = source()

    def version(_):
        damage(learner, field)
        return "fake-mutating-version"

    monkeypatch.setattr(smoke, "version", version)
    result = run(learner, boundary, probe_test_seed=24)
    assert not result["complete"] and not boundary[1]


@pytest.mark.parametrize("target", ["source", "inference"])
@pytest.mark.parametrize("stage", ["observation", "prediction", "history", "step", "work", "close"])
def test_should_refuse_late_owner_replacement_without_erasing_paid_controls(
    boundary, monkeypatch, target, stage
):
    learner = source()
    captured = {}
    original_fork = learner.fork

    def fork():
        captured["inference"] = original_fork()
        return captured["inference"]

    monkeypatch.setattr(learner, "fork", fork)

    def corrupt():
        if "inference" in captured:
            damage(learner if target == "source" else captured["inference"], "features")

    if stage == "observation":
        original = vars(native.ManiSkillPickCubeEpisode)["observation"].fget

        def observed(port):
            value = original(port)
            corrupt()
            return value

        monkeypatch.setattr(native.ManiSkillPickCubeEpisode, "observation", property(observed))
    elif stage == "prediction":
        original = near.NumpyNearestContinuousBCLearner.predict_with_work

        def predict(owner, values):
            value = original(owner, values)
            corrupt()
            return value

        monkeypatch.setattr(near.NumpyNearestContinuousBCLearner, "predict_with_work", predict)
    elif stage == "history":
        original = ActionHistoryEncoding.report

        def history(encoding):
            value = original(encoding)
            corrupt()
            return value

        monkeypatch.setattr(ActionHistoryEncoding, "report", history)
    elif stage == "step":
        before = boundary[0].after_step

        def after():
            before()
            corrupt()

        boundary[0].after_step = after
    elif stage == "work":
        original = vars(native.ManiSkillPickCubeEpisode)["work"].fget

        def work(port):
            value = original(port)
            corrupt()
            return value

        monkeypatch.setattr(native.ManiSkillPickCubeEpisode, "work", property(work))
    else:
        original = native.ManiSkillPickCubeEpisode.close

        def close(port):
            original(port)
            corrupt()

        monkeypatch.setattr(native.ManiSkillPickCubeEpisode, "close", close)
    result = run(learner, boundary, probe_test_seed=24)
    assert not result["complete"]
    assert result["original_owner_exact"] is not True or result["frozen_state_exact"] is not True
    assert boundary[0].close_calls == 1
    paid = result["work"]["completed_steps"]
    assert paid == len(boundary[0].commands)
    assert paid == (2 if stage in ("work", "close") else 1 if stage == "step" else 0)


def test_should_preserve_unknown_failed_control_attempt_and_cleanup(boundary):
    def lost():
        raise RuntimeError("fake control acknowledgement lost")

    boundary[0].after_step = lost
    result = run(source(), boundary, probe_test_seed=24)
    assert not result["complete"] and len(boundary[0].commands) == 1
    assert result["work"]["step_attempts"] == 1 and result["work"]["completed_steps"] == 0
    assert result["work"]["possible_failed_physics_steps"] > 0 and boundary[0].close_calls == 1


def test_should_preserve_exclusive_output_before_work(boundary):
    boundary[2].write_text("retained")
    with pytest.raises(FileExistsError):
        run(source(), boundary, probe_test_seed=24)
    assert boundary[2].read_text() == "retained" and not boundary[1]


@pytest.mark.parametrize("seed", [17, 19])
def test_should_keep_original_nearest_train_routes(boundary, seed):
    result = run(source(), boundary, train_seed=seed)
    assert result["complete"] and result["role"] == "train" and "scoring_only" not in result
    assert boundary[0].reset_calls == [seed]


@pytest.mark.parametrize("family", ["bp", "pc", "neutral-cpc", "active-cpc"])
def test_should_keep_four_existing_neural_reserved_routes_without_rng(boundary, family):
    charge("literal_sources")
    base = bp.ContinuousBCPolicy(cube_taskspace_spec(), hidden_dim=2)
    arrays = (np.zeros((12, 2)), np.zeros(2), np.zeros((2, 4)), np.array((0.1, -0.2, 0.3, -0.4)))
    velocity = cast(bp.Parameters, tuple(np.zeros_like(a) for a in arrays))
    learner: bp.NumpyContinuousBCLearner | pc.NumpyContinuousPCLearner
    if family == "bp":
        learner = object.__new__(bp.NumpyContinuousBCLearner)
        learner._policy = base
        learner._state = bp._snapshot(bp.ContinuousBCSnapshot(base, arrays, velocity, 0, 0), base)
    else:
        chemistry = (
            None
            if family == "pc"
            else CoupledCircadianConfig.matched_pc_control()
            if family == "neutral-cpc"
            else CoupledCircadianConfig()
        )
        pc_policy = pc.ContinuousPCPolicy(base, circadian=chemistry)
        learner = object.__new__(pc.NumpyContinuousPCLearner)
        learner._policy = pc_policy
        learner._state = pc._snapshot(
            pc.ContinuousPCSnapshot(
                pc_policy, arrays, velocity, 0, 0, pc.cpc.initial_continuous_wake(chemistry, 2)
            ),
            pc_policy,
        )
    result = run(learner, boundary, probe_test_seed=24)
    assert result["complete"], result
    assert result["role"] == "probe_test" and result["prediction_returns"] == 2


@pytest.mark.parametrize("fault", ["throw", "erase_both", "erase_lost_attempt"])
def test_should_capture_broken_work_read_without_losing_paid_or_unknown_calls(
    boundary, monkeypatch, fault
):
    original = vars(native.ManiSkillPickCubeEpisode)["work"].fget

    if fault == "erase_lost_attempt":

        def lost():
            raise RuntimeError("lost fake control acknowledgement")

        boundary[0].after_step = lost

    def changed(port):
        if fault == "throw":
            raise RuntimeError("work observation failed after paid controls")
        port._attempts = port._completed = 0
        port._records.clear()
        return original(port)

    monkeypatch.setattr(native.ManiSkillPickCubeEpisode, "work", property(changed))
    result = run(source(), boundary, probe_test_seed=24)
    assert not result["complete"] and boundary[0].close_calls == 1
    attempts, returned = (1, 0) if fault == "erase_lost_attempt" else (2, 2)
    assert result["nearest_control_calls"] == dict(attempted=attempts, returned=returned)
    assert result["nearest_work_scope"] == "outer_call_bounds_native_metadata_untrusted"
    assert result["work"]["step_attempts"] == attempts
    assert result["work"]["completed_steps"] == returned
    assert (result["work"]["possible_failed_physics_steps"] > 0) == (returned == 0)
    assert (
        json.loads(boundary[2].read_text())["nearest_control_calls"]
        == result["nearest_control_calls"]
    )


@pytest.mark.parametrize("fault", ["returned_metadata", "private_counters"])
def test_should_refuse_changed_work_and_keep_paid_control_acknowledgements(
    boundary, monkeypatch, fault
):
    original = vars(native.ManiSkillPickCubeEpisode)["work"].fget

    def changed(port):
        value = original(port)
        if fault == "returned_metadata":
            return replace(
                value,
                step_attempts=0,
                completed_steps=0,
                completed_control_physics_steps=0,
                possible_failed_physics_steps=0,
            )
        port._attempts = port._completed = 0
        return value

    monkeypatch.setattr(native.ManiSkillPickCubeEpisode, "work", property(changed))
    result = run(source(), boundary, probe_test_seed=24)
    assert not result["complete"]
    assert result["nearest_control_calls"] == dict(attempted=2, returned=2)
    assert result["work"]["completed_steps"] == 2
    assert boundary[0].close_calls == 1
