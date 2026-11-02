"""NEW exact retained23 boundary cohort; fake scenes, no SDK/data/model RNG."""

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts import maniskill_continuous_smoke as smoke
from src.adapters import numpy_retained_nearest_bc as retained
from src.adapters.numpy_nearest_continuous_bc import (
    NumpyNearestContinuousBCLearner,
    FixedInputNearestContinuousBCPolicy,
)
from src.core.cube_taskspace_state import cube_taskspace_spec
from src.core.pick_cube_taskspace_history import PickCubeTaskSpaceHistoryRecipe
from src.core.replay_retention import ReplayRetentionPolicy
from src.infra import maniskill_continuous as native
from test_state_quaternion_robot_learning import QuaternionEnvironment
from test_retained_nearest_memory import _process_peak_bytes

WORK: dict[str, int] = {}
PRIOR: dict[str, int] = {}
CAPS = dict(
    initializers=160,
    updates=512,
    incoming_rows=2048,
    forks=1024,
    predictions=1024,
    query_rows=4096,
    candidates=65536,
    coordinates=786432,
    snapshots=32768,
    validators=131072,
    copy_bytes=512 * 1024**2,
    hash_bytes=1024**3,
    creations=80,
    controls=160,
    closes=80,
)


def charge(name, amount=1):
    WORK[name] = WORK.get(name, 0) + amount
    assert WORK[name] + PRIOR.get(name, 0) <= CAPS[name], name


@pytest.fixture(scope="module", autouse=True)
def finite_work():
    PRIOR.update(json.loads(os.environ.get("RETAINED_TRAIN_NATIVE_PRIOR_WORK", "{}")))

    def wrap(method, kind):
        def call(owner, *args, **kwargs):
            charge(kind)
            if kind == "updates":
                charge("incoming_rows", len(args[0]))
            if kind == "predictions":
                rows = len(args[0])
                candidates = rows * len(owner._state.prototype_features)
                charge("query_rows", rows)
                charge("candidates", candidates)
                charge("coordinates", candidates * 12)
            return method(owner, *args, **kwargs)

        return call

    validate = retained.retained_snapshot

    def validated(state, policy, rule):
        charge("validators")
        charge("copy_bytes", state.prototype_features.nbytes + state.prototype_actions.nbytes)
        return validate(state, policy, rule)

    ids = retained._ids

    def hashed(features, actions):
        charge("hash_bytes", features.nbytes + actions.nbytes + 24 * len(features))
        return ids(features, actions)

    with pytest.MonkeyPatch.context() as patch:
        for cls in (NumpyNearestContinuousBCLearner, retained.NumpyRetainedNearestBCLearner):
            for name, kind in (
                ("__init__", "initializers"),
                ("train_batch", "updates"),
                ("fork", "forks"),
                ("predict_with_work", "predictions"),
                ("snapshot_state", "snapshots"),
            ):
                if name in cls.__dict__:
                    patch.setattr(cls, name, wrap(cls.__dict__[name], kind))
        patch.setattr(retained, "retained_snapshot", validated)
        patch.setattr(smoke, "retained_snapshot", validated)
        patch.setattr(retained, "_ids", hashed)
        patch.setattr(np.random, "default_rng", forbidden)
        yield
    peak = _process_peak_bytes()
    target = os.environ.get("RETAINED_TRAIN_NATIVE_RECEIPT")
    if target:
        with Path(target).open("x", encoding="utf8") as handle:
            json.dump(
                dict(
                    work=WORK,
                    aggregate={k: v + PRIOR.get(k, 0) for k, v in WORK.items()},
                    actual_pytest_peak_bytes=peak,
                ),
                handle,
                indent=2,
            )
    assert peak <= 512 * 1024**2


def forbidden(*args, **kwargs):
    pytest.fail("refused request reached source/native/RNG")


def source(*, legacy=False):
    policy = FixedInputNearestContinuousBCPolicy(
        cube_taskspace_spec(), max_prototypes=2 if not legacy else 256
    )
    owner = (
        NumpyNearestContinuousBCLearner(policy)
        if legacy
        else retained.NumpyRetainedNearestBCLearner(policy, ReplayRetentionPolicy("recent_fifo"))
    )
    features = np.zeros((3, 12))
    features[:, 0] = (0.0, 0.01, 0.02)
    owner.train_batch(features, np.tile((0.125, 0.0, 0.0, -1.0), (3, 1)))
    return owner


@pytest.fixture
def boundary(monkeypatch, tmp_path):
    env = QuaternionEnvironment()
    env.after_step = lambda: setattr(env, "bad_reward", np.array([float(len(env.commands) == 2)]))
    old_step, old_close = env.step, env.close

    def step(command):
        charge("controls")
        return old_step(command)

    def close():
        charge("closes")
        return old_close()

    def make():
        charge("creations")
        return env

    monkeypatch.setattr(env, "step", step)
    monkeypatch.setattr(env, "close", close)
    monkeypatch.setattr(native, "_make_environment", make)
    monkeypatch.setattr(smoke, "version", lambda _: "literal-retained23")
    return env, tmp_path / "trace.json"


def run(owner, boundary, **options):
    recipe = options.pop("action_history_recipe", PickCubeTaskSpaceHistoryRecipe())
    return smoke.run_frozen_rollout(
        boundary[1], learner=owner, action_history_recipe=recipe, **options
    ).report


@pytest.mark.parametrize("mode", ["retained23", "retained_train", "legacy23"])
def test_scoring_binds_role_reset_full_state_and_native_work(boundary, mode):
    owner = source(legacy=mode == "legacy23")
    pin = None if mode == "legacy23" else retained.pin_retained_native(owner)
    report = run(owner, boundary, **(dict(probe_train_seed=23) if mode != "retained_train" else {}))
    assert report["complete"], report
    assert report["source_training_rows"] == 3
    assert report["nearest_prototype_rows"] == (3 if mode == "legacy23" else 2)
    assert (
        report["original_owner_exact"]
        is report["frozen_state_exact"]
        is report["action_history_exact"]
        is True
    )
    assert report["optimizer_updates"] == 0 and report["work"]["completed_steps"] == 2
    assert boundary[0].close_calls == 1
    if mode != "retained_train":
        assert report["role"] == "probe_train" and report["seed"] == 23 and report["scoring_only"]
        assert report["episode"]["episode_id"] == "pick-cube-probe-train-smoke-23"
        assert report["nearest_work_scope"] == "validated_native_work"
        assert report["nearest_control_calls"] == dict(attempted=2, returned=2)
    if pin is not None:
        retained.require_retained_native(owner, pin)
        assert report["retained_control_calls"] == dict(attempted=2, returned=2)
    assert json.loads(boundary[1].read_text())["complete"]


@pytest.mark.parametrize(
    "options",
    [
        dict(probe_train_seed=22),
        dict(probe_train_seed=24),
        dict(probe_train_seed="23"),
        dict(probe_train_seed=True),
        dict(probe_train_seed=23, train_seed=19),
        dict(probe_train_seed=23, train_seed=20),
        dict(probe_train_seed=23, probe_test_seed=24),
        dict(probe_train_seed=23, guard_seed=21),
    ],
)
def test_wrong_reset_or_mixed_role_refuses_before_source_pin(monkeypatch, boundary, options):
    owner = source()
    monkeypatch.setattr(smoke, "pin_retained_native", forbidden)
    report = run(owner, boundary, **options)
    assert not report["complete"] and report["creation_attempts"] == 0
    assert boundary[0].reset_calls == []


@pytest.mark.parametrize(
    "options",
    [
        dict(probe_test_seed=24),
        dict(guard_seed=21),
        dict(probe_train_seed=23, visual_observation=True),
        dict(probe_train_seed=23, observation_clock=object()),
        dict(probe_train_seed=23, readiness_corrections=True),
        dict(probe_train_seed=23, corrective_queries=True),
        dict(probe_train_seed=23, observation_projection=object()),
        dict(probe_train_seed=23, action_decoding_recipe=object()),
        dict(probe_train_seed=23, probe_correction_consent=object()),
    ],
)
def test_other_authorities_and_optional_modes_remain_refused(boundary, options):
    report = run(source(), boundary, **options)
    assert not report["complete"] and report["creation_attempts"] == 0


@pytest.mark.parametrize("stage", ["prediction", "step", "work"])
@pytest.mark.parametrize("target", ["source", "inference"])
def test_late_full_state_corruption_refuses_and_preserves_paid_native_work(
    monkeypatch, boundary, stage, target
):
    owner = source()
    captured: dict[str, Any] = {}
    fork = owner.fork

    def forked():
        captured["inference"] = fork()
        return captured["inference"]

    monkeypatch.setattr(owner, "fork", forked)

    def corrupt():
        current = owner if target == "source" else captured["inference"]
        current._state.prototype_actions[0, 0] += 0.25

    original: Any
    if stage == "prediction":
        original = NumpyNearestContinuousBCLearner.predict_with_work

        def predict(*args, **kwargs):
            value = original(*args, **kwargs)
            corrupt()
            return value

        monkeypatch.setattr(NumpyNearestContinuousBCLearner, "predict_with_work", predict)
    elif stage == "step":
        original = native.ManiSkillPickCubeEpisode.step

        def step(*args, **kwargs):
            value = original(*args, **kwargs)
            corrupt()
            return value

        monkeypatch.setattr(native.ManiSkillPickCubeEpisode, "step", step)
    else:
        original = vars(native.ManiSkillPickCubeEpisode)["work"].fget

        def work(port):
            value = original(port)
            corrupt()
            return value

        monkeypatch.setattr(native.ManiSkillPickCubeEpisode, "work", property(work))
    report = run(owner, boundary, probe_train_seed=23)
    assert not report["complete"] and boundary[0].close_calls == 1
    paid = 0 if stage == "prediction" else 1 if stage == "step" else 2
    assert (
        report["nearest_control_calls"]
        == report["retained_control_calls"]
        == dict(attempted=paid, returned=paid)
    )
    assert report["work"]["completed_steps"] == paid


@pytest.mark.parametrize("fault", ["lost_ack", "work_throw", "role_change"])
def test_native_failure_uses_original_outer_call_witness_and_closes(monkeypatch, boundary, fault):
    if fault == "work_throw":

        def work(port):
            raise RuntimeError("literal work getter failed")

        monkeypatch.setattr(native.ManiSkillPickCubeEpisode, "work", property(work))
    else:
        original = native.ManiSkillPickCubeEpisode.step

        def step(port, action):
            if fault == "lost_ack":

                def lost():
                    raise RuntimeError("literal lost native ACK")

                boundary[0].after_step = lost
            value = original(port, action)
            if fault == "role_change":
                port._role = "probe_test"
            return value

        monkeypatch.setattr(native.ManiSkillPickCubeEpisode, "step", step)
    report = run(source(), boundary, probe_train_seed=23)
    assert not report["complete"] and boundary[0].close_calls == 1
    if fault == "lost_ack":
        assert (
            report["nearest_control_calls"]
            == report["retained_control_calls"]
            == dict(attempted=1, returned=0)
        )
        assert (
            report["unreturned_nearest_control_possible"]
            and report["unreturned_retained_control_possible"]
        )
        assert report["work"]["possible_failed_physics_steps"] > 0
    else:
        paid = 2 if fault == "work_throw" else 1
        assert report["work"]["completed_steps"] == paid
        assert report["nearest_control_calls"] == dict(attempted=paid, returned=paid)


def test_aliasing_fork_refuses_before_native_creation(monkeypatch, boundary):
    owner = source()
    fork = owner.fork

    def aliased():
        child = fork()
        object.__setattr__(child._state, "prototype_features", owner._state.prototype_features)
        return child

    monkeypatch.setattr(owner, "fork", aliased)
    report = run(owner, boundary, probe_train_seed=23)
    assert not report["complete"] and report["creation_attempts"] == 0


@pytest.mark.parametrize("fault", ["counts", "role", "seed", "history"])
def test_final_retained_getter_cannot_change_previously_validated_native_counts(
    monkeypatch, boundary, fault
):
    captured: dict[str, Any] = {}
    history = smoke._check_action_history_port

    def observed(port, encoding, report, **kwargs):
        captured.update(port=port, report=report)
        return history(port, encoding, report, **kwargs)

    snapshot = retained.NumpyRetainedNearestBCLearner.snapshot_state

    def getter(owner):
        value = snapshot(owner)
        if captured.get("report", {}).get("nearest_work_scope") == "validated_native_work":
            if fault == "counts":
                captured["port"]._completed += 1
            elif fault == "role":
                captured["port"]._role = "probe_test"
            elif fault == "seed":
                captured["port"]._seed = 24
            else:
                captured["report"]["action_history"]["invalid_after_validation"] = True
        return value

    monkeypatch.setattr(smoke, "_check_action_history_port", observed)
    monkeypatch.setattr(retained.NumpyRetainedNearestBCLearner, "snapshot_state", getter)
    report = run(source(), boundary, probe_train_seed=23)
    assert not report["complete"] and report.get("retained_state_error")
    assert report["nearest_work_scope"] == "outer_call_bounds_native_metadata_untrusted"
    assert report["work"]["completed_steps"] == 2
    assert boundary[0].close_calls == 1
