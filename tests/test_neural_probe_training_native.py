"""New frozen23 neural boundary; manual tiny models, fake scenes, no SGD/SDK."""

from collections import defaultdict
from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest

from scripts import maniskill_continuous_smoke as smoke
from src.adapters import numpy_continuous_bc as bp
from src.adapters import numpy_continuous_pc as pc
from src.core import continuous_bc as core
from src.core.cube_taskspace_state import cube_taskspace_spec
from src.core.coupled_circadian_predictive_coding import CoupledCircadianConfig
from src.core.pick_cube_taskspace_history import PickCubeTaskSpaceHistoryRecipe
from src.core.pick_cube_action_history import PickCubeActionHistoryRecipe
from src.infra import maniskill_continuous as native
from test_state_quaternion_robot_learning import QuaternionEnvironment

WORK: defaultdict[str, int] = defaultdict(int)
CAPS = dict(
    sources=32,
    scenes=16,
    controls=32,
    forwards=64,
    forward_rows=64,
    predictions=64,
    prediction_rows=64,
    snapshots=65536,
    snapshot_bytes=512 * 1024**2,
    forbidden=0,
)


def charge(key, amount=1):
    WORK[key] += amount
    assert WORK[key] <= CAPS[key], "permanent411fixturecap:" + key


def forbidden(*args, **options):
    charge("forbidden")


@pytest.fixture(autouse=True)
def meter(monkeypatch):
    for kind in (bp.NumpyContinuousBCLearner, pc.NumpyContinuousPCLearner):
        for name in ("__init__", "train_batch"):
            monkeypatch.setattr(kind, name, forbidden)
        snap, predict = kind.snapshot_state, kind.predict

        def snapshot(owner, original=snap):
            charge("snapshots")
            value = original(owner)
            charge("snapshot_bytes", sum(a.nbytes for a in value.parameters + value.velocity))
            return value

        def predicted(owner, x, original=predict):
            charge("predictions")
            charge("prediction_rows", len(x))
            return original(owner, x)

        monkeypatch.setattr(kind, "snapshot_state", snapshot)
        monkeypatch.setattr(kind, "predict", predicted)
    forward, step = core._forward, native.ManiSkillPickCubeEpisode.step

    def forwarded(x, *args):
        charge("forwards")
        charge("forward_rows", len(x))
        return forward(x, *args)

    def stepped(owner, action):
        charge("controls")
        return step(owner, action)

    monkeypatch.setattr(core, "_forward", forwarded)
    pc_forward = pc.energy._feedforward

    def forwarded_pc(parameters, values):
        charge("forwards")
        charge("forward_rows", len(values))
        return pc_forward(parameters, values)

    monkeypatch.setattr(pc.energy, "_feedforward", forwarded_pc)
    monkeypatch.setattr(native.ManiSkillPickCubeEpisode, "step", stepped)
    for owner, name in (
        (core, "continuous_bc_loss_and_gradients"),
        (np.random, "default_rng"),
        (np.linalg, "lstsq"),
        (bp.NumpyContinuousBCLearner, "train_full_gauss_newton_batch"),
        (bp.NumpyContinuousBCLearner, "train_head_gauss_newton_batch"),
        (pc.NumpyContinuousPCLearner, "train_batch_with_work"),
        (smoke, "PickCubeCorrectionQuery"),
    ):
        monkeypatch.setattr(owner, name, forbidden)


def source(family="bp", *, role="train"):
    charge("sources")
    base = bp.ContinuousBCPolicy(cube_taskspace_spec(), hidden_dim=2, training_role=role)
    arrays = cast(
        bp.Parameters,
        (np.zeros((12, 2)), np.zeros(2), np.zeros((2, 4)), np.array((0.1, -0.2, 0.3, -0.4))),
    )
    velocity = cast(bp.Parameters, tuple(np.zeros_like(a) for a in arrays))
    owner: bp.NumpyContinuousBCLearner | pc.NumpyContinuousPCLearner
    if family == "bp":
        owner = object.__new__(bp.NumpyContinuousBCLearner)
        owner._policy = base
        owner._state = bp._snapshot(
            bp.ContinuousBCSnapshot(base, arrays, velocity, 80, 17600), base
        )
    else:
        chemistry = (
            None
            if family == "pc"
            else CoupledCircadianConfig.matched_pc_control()
            if family == "neutral-cpc"
            else CoupledCircadianConfig()
        )
        policy = pc.ContinuousPCPolicy(base, circadian=chemistry)
        owner = object.__new__(pc.NumpyContinuousPCLearner)
        owner._policy = policy
        owner._state = pc._snapshot(
            pc.ContinuousPCSnapshot(
                policy, arrays, velocity, 0, 0, pc.cpc.initial_continuous_wake(chemistry, 2)
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
        charge("scenes")
        created.append(True)
        return environment

    monkeypatch.setattr(native, "_make_environment", factory)
    monkeypatch.setattr(smoke, "version", lambda _: "literal-native411")
    return environment, created, tmp_path / "result.json"


def run(owner, boundary, **options):
    return smoke.run_frozen_rollout(
        boundary[2],
        learner=owner,
        action_history_recipe=PickCubeTaskSpaceHistoryRecipe(),
        **options,
    ).report


@pytest.mark.parametrize("family", ["bp", "pc", "neutral-cpc", "active-cpc"])
def test_fixed23_neural_scoring_keeps_private_readout_and_source_frozen(boundary, family):
    owner = source(family)
    before = owner.snapshot_state()
    native_state = owner._state
    report = run(owner, boundary, probe_train_seed=23)
    assert report["complete"], report
    assert report["role"] == report["episode"]["role"] == "probe_train"
    assert report["seed"] == report["probe_train_seed"] == 23 and report["scoring_only"] is True
    assert report["episode"]["episode_id"] == "pick-cube-probe-train-smoke-23"
    assert report["prediction_attempts"] == report["prediction_returns"] == 2
    assert report["optimizer_updates"] == 0 and "corrections" not in report
    assert (
        report["frozen_state_exact"]
        is report["original_owner_exact"]
        is report["action_history_exact"]
        is True
    )
    assert (
        owner._state is native_state and owner._state.completed_updates == before.completed_updates
    )
    for a, b in zip(
        owner._state.parameters + owner._state.velocity,
        before.parameters + before.velocity,
        strict=True,
    ):
        np.testing.assert_array_equal(a, b)
    assert boundary[0].reset_calls == [23] and boundary[0].close_calls == 1


@pytest.mark.parametrize(
    "options,role,seed",
    [(dict(probe_test_seed=24), "probe_test", 24), (dict(train_seed=17), "train", 17)],
)
def test_original_neural_test_and_train_scoring_stay_available(boundary, options, role, seed):
    report = run(source(), boundary, **options)
    assert report["complete"] and report["role"] == role and report["seed"] == seed


@pytest.mark.parametrize(
    "fault", ["private_policy", "wrong_seed", "two_roles", "train19", "corrections", "recipe"]
)
def test_neural23_refuses_unreleased_or_mixed_requests_before_scene(boundary, fault):
    owner = source(role="probe_train" if fault == "private_policy" else "train")
    options: dict[str, Any] = dict(probe_train_seed=23)
    if fault == "wrong_seed":
        options["probe_train_seed"] = 24
    elif fault == "two_roles":
        options["probe_test_seed"] = 24
    elif fault == "train19":
        options["train_seed"] = 19
    elif fault == "corrections":
        options["corrective_queries"] = True
    if fault == "recipe":
        report = smoke.run_frozen_rollout(
            boundary[2],
            learner=owner,
            action_history_recipe=PickCubeActionHistoryRecipe(),
            **options,
        ).report
    else:
        report = run(owner, boundary, **options)
    assert not report["complete"] and report["error"] and not boundary[1]


def test_neural23_refuses_source_mutation_after_paid_control(boundary):
    owner = source()

    def changed():
        owner._state.parameters[0][0, 0] += 0.5

    boundary[0].after_step = changed
    report = run(owner, boundary, probe_train_seed=23)
    assert not report["complete"] and report["error"]
    assert report["prediction_returns"] == 1 and len(boundary[0].commands) == 1
    assert report["optimizer_updates"] == 0 and owner._state.completed_updates == 80


def test_neural23_keeps_lost_native_step_attempt_paid(boundary):
    boundary[0].step_error = RuntimeError("lost physical return")
    report = run(source(), boundary, probe_train_seed=23)
    assert not report["complete"] and report["error"]
    assert report["work"]["step_attempts"] == 1 and report["work"]["completed_steps"] == 0
    assert report["work"]["possible_failed_physics_steps"] == 5
    assert report["prediction_returns"] == 1 and boundary[0].close_calls == 1


@pytest.mark.parametrize("fault", ["throw", "erase"])
def test_neural23_retains_paid_controls_when_native_work_getter_is_broken(
    boundary, monkeypatch, fault
):
    original = vars(native.ManiSkillPickCubeEpisode)["work"].fget

    def broken(port):
        value = original(port)
        if port._completed < 2:
            return value
        if fault == "throw":
            raise RuntimeError("lost work getter")
        return replace(value, step_attempts=0, completed_steps=0, completed_control_physics_steps=0)

    monkeypatch.setattr(native.ManiSkillPickCubeEpisode, "work", property(broken))
    report = run(source(), boundary, probe_train_seed=23)
    assert not report["complete"]
    assert report["work"]["step_attempts"] == report["work"]["completed_steps"] == 2
    assert report["neural_probe_control_calls"] == dict(attempted=2, returned=2)
    assert report["neural_probe_work_scope"] == "outer_call_bounds_native_metadata_untrusted"
    assert boundary[0].close_calls == 1
