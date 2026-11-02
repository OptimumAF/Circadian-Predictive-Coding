"""New literal/fake-port checks for original23 reproduction; no recorded inputs."""

from dataclasses import replace
import json
from typing import Any

import numpy as np
import pytest

from scripts import maniskill_continuous_smoke as smoke
from src.adapters import continuous_probe as probe
from src.adapters import numpy_nearest_continuous_bc as near
from src.adapters import numpy_continuous_bc as bp
from src.adapters import numpy_continuous_pc as pc
from src.app.action_history_encoding import ActionHistoryEncoding
from src.core import continuous_bc_diagnostics as diagnostics
from src.core.continuous_demonstration import ContinuousEpisode, ContinuousTransition
from src.core.cube_taskspace_state import cube_taskspace_spec
from src.core.pick_cube_taskspace_history import PickCubeTaskSpaceHistoryRecipe
from src.infra import maniskill_continuous as native
from test_state_quaternion_robot_learning import QuaternionEnvironment

WORK = dict(
    sources=0,
    neural_owners=0,
    appends=0,
    append_rows=0,
    retrievals=0,
    queries=0,
    candidates=0,
    features=0,
    controls=0,
    creations=0,
    kernels=0,
    pairs=0,
    conversions=0,
    snapshots=0,
    snapshot_bytes=0,
    forbidden=0,
)
CAPS = dict(
    sources=128,
    neural_owners=8,
    appends=64,
    append_rows=128,
    retrievals=256,
    queries=512,
    candidates=1024,
    features=12288,
    controls=256,
    creations=128,
    kernels=64,
    pairs=256,
    conversions=2048,
    snapshots=65536,
    snapshot_bytes=256 * 1024**2,
    forbidden=0,
)


def charge(key, amount=1):
    WORK[key] += amount
    assert WORK[key] <= CAPS[key], "permanent fixture cap: " + key


@pytest.fixture(autouse=True)
def meter(monkeypatch):
    def forbidden(*args, **kwargs):
        charge("forbidden")

    for kind in (
        near.NumpyNearestContinuousBCLearner,
        bp.NumpyContinuousBCLearner,
        pc.NumpyContinuousPCLearner,
    ):
        monkeypatch.setattr(kind, "__init__", forbidden)
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    original_append = near.NumpyNearestContinuousBCLearner.train_batch
    original_predict = near.NumpyNearestContinuousBCLearner.predict_with_work
    original_snapshot = near.NumpyNearestContinuousBCLearner.snapshot_state
    original_step = native.ManiSkillPickCubeEpisode.step
    original_kernel = diagnostics.continuous_input_target_consistency

    def append(owner, features, targets):
        charge("appends")
        charge("append_rows", len(features))
        return original_append(owner, features, targets)

    def predict(owner, features):
        charge("retrievals")
        charge("queries", len(features))
        count = len(features) * owner._state.training_examples
        charge("candidates", count)
        charge("features", count * 12)
        return original_predict(owner, features)

    def snapshot(owner):
        charge("snapshots")
        state = original_snapshot(owner)
        charge("snapshot_bytes", state.prototype_features.nbytes + state.prototype_actions.nbytes)
        return state

    def step(port, action):
        charge("controls")
        return original_step(port, action)

    def kernel(features, targets, spec, **kwargs):
        charge("kernels")
        charge("pairs", len(features) * (len(features) - 1) // 2)
        charge("conversions", len(features) * 4)
        return original_kernel(features, targets, spec, **kwargs)

    monkeypatch.setattr(near.NumpyNearestContinuousBCLearner, "train_batch", append)
    monkeypatch.setattr(near.NumpyNearestContinuousBCLearner, "predict_with_work", predict)
    monkeypatch.setattr(near.NumpyNearestContinuousBCLearner, "snapshot_state", snapshot)
    monkeypatch.setattr(native.ManiSkillPickCubeEpisode, "step", step)
    monkeypatch.setattr(diagnostics, "continuous_input_target_consistency", kernel)
    monkeypatch.setattr(native, "_make_environment", forbidden)
    monkeypatch.setattr(smoke, "PickCubeCorrectionQuery", forbidden)


def source():
    charge("sources")
    policy = near.FixedInputNearestContinuousBCPolicy(cube_taskspace_spec())
    owner = object.__new__(near.NumpyNearestContinuousBCLearner)
    owner._policy = policy
    owner._state = near._snapshot(
        near.NearestContinuousBCSnapshot(
            policy, np.zeros((2, 12)), np.tile((-1.0, 0.0, 0.0, 1.0), (2, 1)), 1, 2
        ),
        policy,
    )
    return owner


@pytest.fixture
def boundary(tmp_path, monkeypatch):
    envs = []

    def factory():
        charge("creations")
        env = QuaternionEnvironment()

        def sparse():
            env.bad_reward = np.array([float(len(env.commands) == 2)])

        env.after_step = sparse
        envs.append(env)
        return env

    monkeypatch.setattr(native, "_make_environment", factory)
    monkeypatch.setattr(smoke, "version", lambda _: "fake-native")
    return envs, tmp_path


def run(owner, boundary, name="trace", **options):
    return smoke.run_frozen_rollout(
        boundary[1] / (name + ".json"),
        learner=owner,
        action_history_recipe=PickCubeTaskSpaceHistoryRecipe(),
        **options,
    ).report


@pytest.mark.parametrize(
    "options,role,seed",
    [
        (dict(probe_train_seed=23), "probe_train", 23),
        (dict(probe_test_seed=24), "probe_test", 24),
        (dict(train_seed=17), "train", 17),
        (dict(train_seed=19), "train", 19),
    ],
)
def test_should_bind_original_role_reset_and_history_with_frozen_private_owner(
    boundary, options, role, seed
):
    owner = source()
    identity = smoke._pin_nearest_probe(owner)
    result = run(owner, boundary, **options)
    assert result["complete"], result
    assert result["role"] == result["episode"]["role"] == role and result["seed"] == seed
    assert (
        result["frozen_state_exact"]
        is result["original_owner_exact"]
        is result["action_history_exact"]
        is True
    )
    assert result["optimizer_updates"] == 0 and result["work"]["completed_steps"] == 2
    assert boundary[0][0].reset_calls == [seed] and boundary[0][0].close_calls == 1
    assert all(a.dtype == np.dtype("float32") for a in boundary[0][0].commands)
    if seed == 23:
        assert result["scoring_only"] is True
        assert result["episode"]["episode_id"] == "pick-cube-probe-train-smoke-23"
        assert result["episode"]["episode_id"] != "pick-cube-probe-train-teacher-23"
        assert result["nearest_work_scope"] == "validated_native_work"
    smoke._require_nearest_probe(identity)


@pytest.mark.parametrize("seed", [24, 29, 17, 31, 37, True, 23.0, "23"])
def test_should_refuse_wrong_training_reset_before_source_getters_or_scene(
    boundary, monkeypatch, seed
):
    owner = source()
    monkeypatch.setattr(owner, "snapshot_state", lambda: pytest.fail("source observed"))
    result = run(owner, boundary, probe_train_seed=seed)
    assert not result["complete"] and result["creation_attempts"] == 0 and not boundary[0]


@pytest.mark.parametrize("extra", [dict(probe_test_seed=24), dict(guard_seed=31)])
def test_should_refuse_mixed_reserved_rights_before_scene(boundary, extra):
    result = run(source(), boundary, probe_train_seed=23, **extra)
    assert not result["complete"] and not boundary[0]


@pytest.mark.parametrize("kind", [bp.NumpyContinuousBCLearner, pc.NumpyContinuousPCLearner])
def test_should_not_grant_neural_models_new_training_reset_rights(boundary, kind):
    charge("neural_owners")
    result = run(object.__new__(kind), boundary, probe_train_seed=23)
    assert not result["complete"] and not boundary[0]


@pytest.mark.parametrize(
    "option",
    [
        "actor",
        "observation_clock",
        "corrective_queries",
        "monitor_joint_positions",
        "mass_context",
        "camera_context",
    ],
)
def test_should_refuse_optional_modes_before_scene(boundary, option):
    value = True if option in ("corrective_queries", "monitor_joint_positions") else object()
    result = run(source(), boundary, probe_train_seed=23, **{option: value})
    assert not result["complete"] and not boundary[0]


@pytest.mark.parametrize("field", ["counter", "policy", "spec", "features", "actions"])
def test_should_refuse_counter_or_equal_binding_changes_before_creation(
    boundary, monkeypatch, field
):
    owner = source()

    def version(_):
        if field == "counter":
            object.__setattr__(owner._state, "completed_updates", 2)
        elif field == "policy":
            object.__setattr__(owner._state, "policy", replace(owner._state.policy))
        elif field == "spec":
            object.__setattr__(owner._state.policy, "spec", replace(owner._state.policy.spec))
        else:
            name = "prototype_" + field
            object.__setattr__(owner._state, name, getattr(owner._state, name).copy())
        return "fake-mutating-version"

    monkeypatch.setattr(smoke, "version", version)
    result = run(owner, boundary, probe_train_seed=23)
    assert not result["complete"] and not boundary[0]


@pytest.mark.parametrize("target", ["source", "inference"])
@pytest.mark.parametrize("stage", ["history", "prediction", "step", "work", "close"])
def test_should_refuse_late_replacements_and_preserve_paid_controls(
    boundary, monkeypatch, target, stage
):
    owner = source()
    captured = {}
    original_fork = owner.fork

    def fork():
        captured["inference"] = original_fork()
        return captured["inference"]

    monkeypatch.setattr(owner, "fork", fork)

    def corrupt():
        if "inference" in captured:
            model = owner if target == "source" else captured["inference"]
            object.__setattr__(
                model._state, "prototype_features", model._state.prototype_features.copy()
            )

    original: Any
    if stage == "history":
        original = ActionHistoryEncoding.report

        def history(encoding):
            value = original(encoding)
            corrupt()
            return value

        monkeypatch.setattr(ActionHistoryEncoding, "report", history)
    elif stage == "prediction":
        original = near.NumpyNearestContinuousBCLearner.predict_with_work

        def predict(*args, **kwargs):
            value = original(*args, **kwargs)
            corrupt()
            return value

        monkeypatch.setattr(near.NumpyNearestContinuousBCLearner, "predict_with_work", predict)
    elif stage == "step":
        original = native.ManiSkillPickCubeEpisode.step

        def step(*args, **kwargs):
            value = original(*args, **kwargs)
            corrupt()
            return value

        monkeypatch.setattr(native.ManiSkillPickCubeEpisode, "step", step)
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
    result = run(owner, boundary, probe_train_seed=23)
    assert not result["complete"]
    assert result["original_owner_exact"] is not True or result["frozen_state_exact"] is not True
    paid = 2 if stage in ("work", "close") else 1 if stage == "step" else 0
    assert result["work"]["completed_steps"] == paid
    assert boundary[0][0].close_calls == 1


@pytest.mark.parametrize("fault", ["lost_ack", "work_throw", "role_change"])
def test_should_capture_failed_native_boundary_without_erasing_work(boundary, monkeypatch, fault):
    if fault == "work_throw":

        def work(port):
            raise RuntimeError("fake work read failed")

        monkeypatch.setattr(native.ManiSkillPickCubeEpisode, "work", property(work))
    else:
        original = native.ManiSkillPickCubeEpisode.step

        def step(port, action):
            if fault == "lost_ack":

                def lost():
                    raise RuntimeError("fake control ACK lost")

                boundary[0][0].after_step = lost
            result = original(port, action)
            if fault == "role_change":
                port._role = "probe_test"
            return result

        monkeypatch.setattr(native.ManiSkillPickCubeEpisode, "step", step)
    result = run(source(), boundary, probe_train_seed=23)
    assert not result["complete"] and boundary[0][0].close_calls == 1
    if fault == "lost_ack":
        assert result["nearest_control_calls"] == dict(attempted=1, returned=0)
        assert result["work"]["possible_failed_physics_steps"] > 0
    else:
        paid = 2 if fault == "work_throw" else 1
        assert result["nearest_control_calls"] == dict(attempted=paid, returned=paid)
        assert result["work"]["completed_steps"] == paid
    assert json.loads((boundary[1] / "trace.json").read_text())["complete"] is False


def packet():
    return ContinuousEpisode(
        "literal-probe23",
        "literal-fresh",
        "probe_train",
        cube_taskspace_spec(),
        (
            ContinuousTransition(
                0,
                (0.0, 10.0) + (0.0,) * 10,
                (1.0, 0.0, 0.0, 1.0),
                0.0,
                (0.0, 11.0) + (0.0,) * 10,
                True,
                False,
            ),
        ),
    )


def test_should_forward_guarded_native_readouts_at_both_training_endpoints(boundary):
    owner = source()
    seen = []

    def observer(readout, phase, guard):
        def observe():
            result = run(readout, boundary, name=phase, probe_train_seed=23)
            seen.append(result)
            if not result["complete"]:
                raise RuntimeError("native readout incomplete")

        guard(observe)

    result = probe.run_nearest_training_probe(owner, packet(), endpoint_observer=observer)
    assert result["complete"], result
    assert len(seen) == 2 and all(r["complete"] for r in seen)
    assert [r["source_training_rows"] for r in seen] == [2, 3]
    assert result["work"]["native_returns"] == 1
    assert owner._state.training_examples == 2


@pytest.mark.parametrize("target", ["source", "readout"])
def test_should_refuse_equal_nested_policy_replacement_from_training_observer(target):
    owner = source()

    def observer(readout, phase, guard):
        model = owner if target == "source" else readout
        object.__setattr__(model._state, "policy", replace(model._state.policy))
        guard()

    result = probe.run_nearest_training_probe(owner, packet(), endpoint_observer=observer)
    assert not result["complete"] and result["work"]["training_attempts"] == 0


def test_should_keep_exclusive_output_and_refuse_invalid_observer_before_work(boundary):
    path = boundary[1] / "trace.json"
    path.write_text("retained")
    with pytest.raises(FileExistsError):
        run(source(), boundary, probe_train_seed=23)
    assert path.read_text() == "retained" and not boundary[0]
    with pytest.raises(ValueError, match="callable"):
        probe.run_nearest_training_probe(source(), packet(), endpoint_observer=object())


@pytest.mark.parametrize("seed", [31, 37])
def test_should_keep_existing_nearest_guard_rights_refused(boundary, seed):
    result = run(source(), boundary, guard_seed=seed)
    assert not result["complete"] and not boundary[0]


def test_should_refuse_zero_history_on_new_training_readout(boundary):
    result = smoke.run_frozen_rollout(
        boundary[1] / "zero.json",
        learner=source(),
        probe_train_seed=23,
        action_history_recipe=PickCubeTaskSpaceHistoryRecipe(mode="zero"),
    ).report
    assert not result["complete"] and not boundary[0]


def test_should_stop_before_append_after_incomplete_native_training_observation(
    boundary, monkeypatch
):
    def work(port):
        raise RuntimeError("fake paid work getter failure")

    monkeypatch.setattr(native.ManiSkillPickCubeEpisode, "work", property(work))
    captured = []

    def observer(readout, phase, guard):
        def observe():
            result = run(readout, boundary, name=phase, probe_train_seed=23)
            captured.append(result)
            if not result["complete"]:
                raise RuntimeError("preceding native observation incomplete")

        guard(observe)

    result = probe.run_nearest_training_probe(source(), packet(), endpoint_observer=observer)
    assert not result["complete"] and result["work"]["training_attempts"] == 0
    assert len(captured) == 1 and captured[0]["nearest_control_calls"] == dict(
        attempted=2, returned=2
    )
    assert captured[0]["work"]["completed_steps"] == 2 and boundary[0][0].close_calls == 1
