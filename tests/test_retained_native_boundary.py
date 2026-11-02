"""New literal retained-state/native-boundary cohort; fake IO, no SDK/data."""

from collections import defaultdict
from dataclasses import replace
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts import maniskill_continuous_smoke as pick
from scripts import push_cube_continuation_study as push
from src.adapters.numpy_nearest_continuous_bc import FixedInputNearestContinuousBCPolicy
from src.adapters.numpy_retained_nearest_bc import (
    NumpyRetainedNearestBCLearner,
    pin_retained_native,
    require_retained_native,
)
from src.core.cube_taskspace_state import cube_taskspace_spec
from src.core.pick_cube_taskspace_history import PickCubeTaskSpaceHistoryRecipe
from src.core.replay_retention import ReplayRetentionPolicy
from src.infra import maniskill_continuous as pick_native
from src.infra import maniskill_push_cube as push_native
from test_state_quaternion_robot_learning import QuaternionEnvironment
from test_push_cube_continuous import FakePushEnvironment
from test_retained_nearest_memory import _process_peak_bytes

WORK = dict(
    initializers=0,
    updates=0,
    rows=0,
    forks=0,
    predictions=0,
    queries=0,
    candidates=0,
    coordinates=0,
    creations=0,
    closes=0,
    controls=0,
    snapshots=0,
    snapshot_bytes=0,
)
CASES: dict[str, Any] = {}


@pytest.fixture(scope="module", autouse=True)
def finite_work():
    prior = json.loads(os.environ.get("RETAINED_NATIVE_PRIOR_WORK", "{}"))
    caps = dict(
        initializers=160,
        updates=160,
        rows=640,
        forks=256,
        predictions=512,
        queries=512,
        candidates=2048,
        coordinates=24576,
        creations=128,
        closes=128,
        controls=256,
        snapshots=65536,
        snapshot_bytes=256 * 1024**2,
    )

    def charge(name, amount=1):
        WORK[name] += amount
        assert WORK[name] + prior.get(name, 0) <= caps[name], name

    def wrap(method, kind):
        def counted(owner, *args, **kwargs):
            charge(kind)
            if kind == "updates":
                charge("rows", len(args[0]))
            if kind == "predictions":
                rows = len(args[0])
                candidates = rows * len(owner._state.prototype_features)
                charge("queries", rows)
                charge("candidates", candidates)
                charge("coordinates", candidates * owner._policy.spec.observation_dim)
            value = method(owner, *args, **kwargs)
            if kind == "snapshots":
                charge(
                    "snapshot_bytes",
                    value.prototype_features.nbytes + value.prototype_actions.nbytes,
                )
            return value

        return counted

    with pytest.MonkeyPatch.context() as patch:
        for name, kind in (
            ("__init__", "initializers"),
            ("train_batch", "updates"),
            ("fork", "forks"),
            ("snapshot_state", "snapshots"),
            ("predict_with_work", "predictions"),
        ):
            patch.setattr(
                NumpyRetainedNearestBCLearner,
                name,
                wrap(getattr(NumpyRetainedNearestBCLearner, name), kind),
            )
        patch.setattr(np.random, "default_rng", lambda *a, **k: pytest.fail("model RNG forbidden"))
        yield charge
    peak = _process_peak_bytes()
    receipt = os.environ.get("RETAINED_NATIVE_RECEIPT")
    if receipt:
        with Path(receipt).open("x", encoding="utf8") as handle:
            json.dump(
                dict(
                    work=WORK,
                    aggregate={n: v + prior.get(n, 0) for n, v in WORK.items()},
                    peak_bytes=peak,
                    cases=CASES,
                ),
                handle,
                indent=2,
                allow_nan=False,
            )
    assert peak <= 512 * 1024**2


def source():
    owner = NumpyRetainedNearestBCLearner(
        FixedInputNearestContinuousBCPolicy(cube_taskspace_spec(), max_prototypes=2),
        ReplayRetentionPolicy("recent_fifo"),
    )
    features = np.zeros((3, 12))
    features[:, 0] = (0.0, 0.01, 0.02)
    owner.train_batch(features, np.tile((0.125, 0.0, 0.0, -1.0), (3, 1)))
    return owner


@pytest.fixture
def boundary(monkeypatch, finite_work, tmp_path):
    pick_env = QuaternionEnvironment()
    pick_env.after_step = lambda: setattr(
        pick_env, "bad_reward", np.array([float(len(pick_env.commands) == 2)])
    )
    counts: defaultdict[str, int] = defaultdict(int)
    push_env = FakePushEnvironment(
        counts, horizon=2, terminated=False, truncated=True, success=False
    )
    for environment in (pick_env, push_env):
        old_step, old_close = environment.step, environment.close

        def step(command, original=old_step):
            finite_work("controls")
            return original(command)

        def close(original=old_close):
            finite_work("closes")
            return original()

        monkeypatch.setattr(environment, "step", step)
        monkeypatch.setattr(environment, "close", close)

    def factory(environment):
        finite_work("creations")
        return environment

    monkeypatch.setattr(pick_native, "_make_environment", lambda: factory(pick_env))
    monkeypatch.setattr(push_native, "_make_environment", lambda: factory(push_env))
    monkeypatch.setattr(pick, "version", lambda _: "fake-retained-boundary")
    return pick_env, push_env, tmp_path / "pick.json"


def run_pick(owner, boundary, **options):
    return pick.run_frozen_rollout(
        boundary[2],
        learner=owner,
        action_history_recipe=PickCubeTaskSpaceHistoryRecipe(),
        **options,
    ).report


@pytest.mark.parametrize("task", ["pick", "push"])
def test_should_serve_retained_rows_and_preserve_cumulative_exposures(task, boundary):
    owner = source()
    pin = pin_retained_native(owner)
    report = run_pick(owner, boundary) if task == "pick" else push.roll_push(owner, label="literal")
    assert report["complete"] is True, report
    assert report["source_training_rows"] == 3
    assert report["nearest_prototype_rows"] == 2
    assert report["nearest_distance_candidates_completed"] == 4
    assert report["prediction_attempts"] == report["prediction_returns"] == 2
    assert (boundary[0] if task == "pick" else boundary[1]).close_calls == 1
    require_retained_native(owner, pin)
    CASES[task] = dict(
        complete=report["complete"],
        source_rows=3,
        stored_rows=2,
        predictions=2,
        candidates=4,
        work=report["work"],
    )


@pytest.mark.parametrize(
    "damage", ["ids", "features", "actions", "state", "policy", "retention", "counter"]
)
def test_should_refuse_native_identity_value_or_counter_corruption(damage):
    owner = source()
    pin = pin_retained_native(owner)
    if damage == "ids":
        owner._state = replace(
            owner._state, prototype_ids=tuple(reversed(owner._state.prototype_ids))
        )
    elif damage == "features":
        owner._state.prototype_features[0, 0] = 123
    elif damage == "actions":
        object.__setattr__(owner._state, "prototype_actions", owner._state.prototype_actions.copy())
    elif damage == "state":
        owner._state = owner.snapshot_state()
    elif damage == "policy":
        owner._policy = replace(owner._policy)
    elif damage == "retention":
        owner._retention = ReplayRetentionPolicy("seeded_reservoir", 17)
    else:
        object.__setattr__(
            owner._state, "selection_candidates", owner._state.selection_candidates + 1
        )
    with pytest.raises(ValueError):
        require_retained_native(owner, pin)


def test_should_close_public_getter_boundary_from_original_native_owners(monkeypatch):
    owner = source()
    pin = pin_retained_native(owner)
    snapshot = NumpyRetainedNearestBCLearner.snapshot_state

    def changed(current):
        value = snapshot(current)
        current._state.prototype_features[0, 0] = 123
        return value

    monkeypatch.setattr(NumpyRetainedNearestBCLearner, "snapshot_state", changed)
    with pytest.raises(ValueError):
        require_retained_native(owner, pin)


@pytest.mark.parametrize(
    "options",
    [
        dict(probe_test_seed=24),
        dict(probe_train_seed=23),
        dict(guard_seed=31),
        dict(corrective_queries=True),
        dict(visual_observation=True),
    ],
)
def test_should_refuse_reserved_or_optional_modes_before_native_creation(options, boundary):
    before = WORK["creations"]
    report = run_pick(source(), boundary, **options)
    assert report["complete"] is False
    assert WORK["creations"] == before


@pytest.mark.parametrize("task", ["pick", "push"])
def test_should_refuse_source_changes_after_paid_control_and_still_close(task, boundary):
    owner = source()
    environment = boundary[0] if task == "pick" else boundary[1]
    environment.after_step = lambda: object.__setattr__(owner._state, "duplicate_rows", 1)
    report = run_pick(owner, boundary) if task == "pick" else push.roll_push(owner, label="corrupt")
    assert report["complete"] is False
    assert environment.close_calls == 1
    assert report["work"]["completed_steps"] == 1
    if task == "pick":
        assert report["retained_control_calls"] == dict(attempted=1, returned=1)
    else:
        assert report["control_returns"] == 1


@pytest.mark.parametrize("task", ["pick", "push"])
def test_should_capture_lost_control_return_and_close(task, boundary):
    def lost():
        raise RuntimeError("lost fake native return")

    environment = boundary[0] if task == "pick" else boundary[1]
    environment.after_step = lost
    report = (
        run_pick(source(), boundary) if task == "pick" else push.roll_push(source(), label="lost")
    )
    assert report["complete"] is False and environment.close_calls == 1
    if task == "pick":
        assert report["retained_control_calls"] == dict(attempted=1, returned=0)
        assert report["unreturned_retained_control_possible"] is True
    else:
        assert report["control_attempts"] == 1 and report["control_returns"] == 0
        assert report["unreturned_native_work_unknown"] is True
    CASES[task + "-lost"] = dict(complete=False, work=report["work"])


def test_should_refuse_replaced_preceding_push_prediction_receipt(monkeypatch, boundary):
    owner = source()
    predict = NumpyRetainedNearestBCLearner.predict_with_work
    previous: list[Any] = []

    def returned(current, values):
        if previous:
            previous[0].actions[0, 0] = 0.5
        value = predict(current, values)
        previous.append(value)
        return value

    monkeypatch.setattr(NumpyRetainedNearestBCLearner, "predict_with_work", returned)
    report = push.roll_push(owner, label="prior")
    assert report["complete"] is False
    assert report["control_returns"] == 1 and boundary[1].close_calls == 1
    assert len(report["nearest_prediction_reports"]) == 2


def test_should_refuse_aliased_inference_fork_before_native_work(monkeypatch, boundary):
    owner = source()
    monkeypatch.setattr(NumpyRetainedNearestBCLearner, "fork", lambda current: current)
    before = WORK["creations"]
    report = run_pick(owner, boundary)
    assert report["complete"] is False and WORK["creations"] == before


@pytest.mark.parametrize("task", ["pick", "push"])
def test_should_refuse_unlisted_subclasses_before_native_creation(task, boundary):
    class Child(NumpyRetainedNearestBCLearner):
        pass

    original = source()
    child = object.__new__(Child)
    child._policy, child._retention, child._state = (
        original._policy,
        original._retention,
        original._state,
    )
    before = WORK["creations"]
    if task == "pick":
        assert run_pick(child, boundary)["complete"] is False
    else:
        with pytest.raises(ValueError):
            push.roll_push(child, label="subclass")
    assert WORK["creations"] == before


def test_should_close_both_native_owners_after_last_inference_getter(monkeypatch, boundary):
    owner = source()
    snapshot = NumpyRetainedNearestBCLearner.snapshot_state

    def deceptive(current):
        value = snapshot(current)
        if current is not owner:
            object.__setattr__(
                owner._state, "selection_candidates", owner._state.selection_candidates + 1
            )
        return value

    def intercept(*args, **kwargs):
        monkeypatch.setattr(NumpyRetainedNearestBCLearner, "snapshot_state", deceptive)
        # This directly tests the shared guard's closing pass. A later outer
        # check does not satisfy this refusal before the next native command.
        with pytest.raises(ValueError):
            kwargs["frozen_guard"]()
        raise RuntimeError("cross-owner refusal observed")

    monkeypatch.setattr(pick, "_rollout", intercept)
    report = run_pick(owner, boundary)
    assert report["complete"] is False and boundary[0].commands == []
    assert boundary[0].close_calls == 1
