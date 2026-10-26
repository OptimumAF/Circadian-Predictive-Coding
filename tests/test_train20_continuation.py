"""One tiny actual TRAIN20 continuation and pre-admission refusal boundaries.

Budget: one H2/D12 initializer (32 normals), one GN update/two training rows,
four axis solves/twenty augmented rows; no explicit prediction, SGD, fit or SDK.
Artifact import loads only this stage's helper, never a historical producer.
"""

from copy import deepcopy
from dataclasses import asdict, replace
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest

from src.adapters import numpy_continuous_bc as bp
from src.core import pick_cube_action_history as history
from src.core.observation_standardization import ObservationStandardization
from src.core.pick_cube_taskspace_history import PickCubeTaskSpaceHistoryRecipe
from test_joint_readiness_corrections import assert_state
from test_pick_cube_action_history import episode


def forbidden(*args, **kwargs):
    pytest.fail("invalid continuation admitted private work or a forbidden operation")


@pytest.fixture(scope="module")
def continuation():
    path = (
        Path(__file__).resolve().parents[1]
        / "artifacts/runs/train20-corrected-continuation-20261021/continuation.py"
    )
    spec = importlib.util.spec_from_file_location("test_train20_continuation_helper", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


@pytest.fixture(scope="module", autouse=True)
def ledger():
    counts = dict(
        initializations=0,
        normal_draws=0,
        GN_attempts=0,
        GN_returns=0,
        training_rows=0,
        solve_attempts=0,
        solve_completed=0,
        augmented_rows=0,
        inputs_requested=False,
    )
    initialize = bp.NumpyContinuousBCLearner.__init__
    train = bp.NumpyContinuousBCLearner.train_head_gauss_newton_batch

    def initialized(owner, policy):
        counts["initializations"] += 1
        counts["normal_draws"] += policy.hidden_dim * (policy.spec.observation_dim + 4)
        return initialize(owner, policy)

    def trained(owner, features, targets):
        counts["GN_attempts"] += 1
        counts["training_rows"] += len(features)
        try:
            result = train(owner, features, targets)
        except Exception as error:
            work = getattr(error, "work", {})
            counts["solve_attempts"] += work.get("solve_attempts", 0)
            counts["solve_completed"] += work.get("solve_completed", 0)
            counts["augmented_rows"] += work.get("solve_rows", 0)
            raise
        counts["GN_returns"] += 1
        counts["solve_attempts"] += result.work["solve_attempts"]
        counts["solve_completed"] += result.work["solve_completed"]
        counts["augmented_rows"] += result.work["solve_rows"]
        return result

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(bp.NumpyContinuousBCLearner, "__init__", initialized)
        patch.setattr(bp.NumpyContinuousBCLearner, "train_head_gauss_newton_batch", trained)
        patch.setattr(bp.NumpyContinuousBCLearner, "train_batch", forbidden)
        patch.setattr(bp.NumpyContinuousBCLearner, "predict", forbidden)
        patch.setattr(bp, "fit_training_standardization", forbidden)
        yield counts
    if os.environ.get("C20_LEDGER"):
        with Path(os.environ["C20_LEDGER"]).open("x", encoding="utf8") as handle:
            json.dump(counts, handle, sort_keys=True)
            handle.write("\n")
    expected_initializations = int(counts["inputs_requested"])
    assert counts["initializations"] == expected_initializations
    assert counts["normal_draws"] == 32 * expected_initializations
    if not counts["inputs_requested"]:
        assert counts["GN_attempts"] == counts["GN_returns"] == counts["training_rows"] == 0
    assert counts["GN_attempts"] <= 1 and counts["training_rows"] <= 2
    assert counts["solve_attempts"] <= 4 and counts["augmented_rows"] <= 20


@pytest.fixture(scope="module")
def inputs(ledger):
    ledger["inputs_requested"] = True
    base = episode(20)
    rows = base.transitions[:2]
    raw = replace(
        base,
        episode_id="pick-cube-observed-grasp-teacher-20",
        context="pick-cube-train-20",
        transitions=(rows[0], replace(rows[1], terminated=True)),
    )
    model = history.action_history_training_episode(raw, PickCubeTaskSpaceHistoryRecipe())
    source = bp.NumpyContinuousBCLearner(bp.ContinuousBCPolicy(model.spec, hidden_dim=2, seed=17))
    assert ledger["initializations"] == 1 and ledger["normal_draws"] == 32
    return source, raw, model


def test_should_publish_one_detached_native_update_with_original_clocks_and_frozen_source(
    continuation, inputs, monkeypatch
):
    source, _, model = inputs
    source = source.fork()
    original = source.snapshot_state()
    original_model = deepcopy(asdict(model))
    runtime_factory = continuation.ActorShadowRuntime
    captured: dict[str, Any] = {}

    def runtime(*args, **options):
        owner = runtime_factory(*args, **options)
        captured.update(runtime=owner, budget=options["budget"], clock=options["clock"])
        return owner

    monkeypatch.setattr(continuation, "ActorShadowRuntime", runtime)
    result = continuation.continue_training(source, model, updates=1)
    report = result.report
    assert report["complete"] is True, report.get("error")
    assert type(result.candidate) is bp.NumpyContinuousBCLearner and result.candidate is not source
    final = result.candidate.snapshot_state()
    assert (final.completed_updates, final.training_examples) == (1, 2)
    assert_state(original, source.snapshot_state())
    assert_state(original, captured["runtime"].actor.snapshot_state().state)
    assert report["source_exact"] is report["actor_exact"] is True
    assert asdict(model) == original_model
    assert report["initial_counts"] == dict(updates=0, training_rows=0)
    assert report["final_counts"] == dict(updates=1, training_rows=2)
    assert report["original_episode"] == dict(
        episode_id=model.episode_id, context=model.context, role="train", rows=2
    )
    assert len(report["native_updates"]) == len(report["applied_updates"]) == 1
    assert (
        report["native_updates"]
        == report["sealed_native_updates"]
        == report["validated_native_updates"]
    )
    native = report["native_updates"][0]
    receipt = report["applied_updates"][0]
    assert native["training_rows"] == 2
    assert native["work"]["solve_completed"] == 4 and native["work"]["solve_rows"] == 20
    assert native["loss_after"] < native["diagnostic"]["value"]
    assert receipt["diagnostic"] == native["diagnostic"]
    assert (receipt["observed_at"], receipt["arrived_at"], receipt["applied_at"]) == (
        1000,
        1001,
        1001,
    )
    assert receipt["episode_id"] == model.episode_id
    assert report["events"] == [
        dict(
            index=0,
            observed_at=1000,
            arrived_at=1001,
            deferred_exact=True,
            update_attempted=True,
            success_validated=True,
        )
    ]
    assert report["progress"]["updates_completed"] == captured["budget"].updates_completed == 1
    assert report["progress"]["replay_examples_completed"] == 0
    assert report["progress"]["logical_tick"] == 1001
    assert report["budget"]["max_training_updates"] == 1
    assert captured["clock"].now() == 1001
    # Caller-owned diagnostic changes cannot alter a detached returned learner.
    report["native_updates"][0]["diagnostic"]["value"] += 1.0
    assert_state(final, result.candidate.snapshot_state())


@pytest.mark.parametrize("updates", [0, 9, True, 1.0, None, "1"])
def test_should_refuse_invalid_update_limit_before_private_runtime(
    continuation, inputs, monkeypatch, updates
):
    source, _, model = inputs
    monkeypatch.setattr(continuation, "ActorShadowRuntime", forbidden)
    monkeypatch.setattr(source, "fork", forbidden)
    with pytest.raises(ValueError):
        continuation.continue_training(source, model, updates=updates)


@pytest.mark.parametrize(
    "fault", ["role", "context", "identity", "spec", "history", "zero_history", "foreign"]
)
def test_should_refuse_nonoriginal_train20_or_corrupt_executed_history_before_work(
    continuation, inputs, monkeypatch, fault
):
    source, raw, model = inputs
    model = deepcopy(model)
    if fault == "role":
        model = replace(model, role="probe_train")
    elif fault == "context":
        model = replace(model, context="pick-cube-train-19")
    elif fault == "identity":
        model = replace(model, episode_id="pick-cube-observed-grasp-teacher-19")
    elif fault == "spec":
        object.__setattr__(model.spec, "observation_dim", 11)
    elif fault == "history":
        row = model.transitions[-1]
        object.__setattr__(row, "next_observation", (*row.next_observation[:-5], *(0.0,) * 5))
    elif fault == "zero_history":
        model = history.action_history_training_episode(
            raw, PickCubeTaskSpaceHistoryRecipe(mode="zero")
        )
    else:
        model = object()
    monkeypatch.setattr(continuation, "ActorShadowRuntime", forbidden)
    monkeypatch.setattr(source, "fork", forbidden)
    with pytest.raises(ValueError):
        continuation.continue_training(source, model, updates=1)


@pytest.mark.parametrize("fault", ["foreign", "role", "standardization", "mixed_objective"])
def test_should_refuse_unsupported_source_before_runtime(continuation, inputs, monkeypatch, fault):
    source, _, model = inputs
    source = source.fork()
    if fault == "foreign":
        source = object()
    else:
        state = source.snapshot_state()
        if fault == "role":
            policy = replace(state.policy, training_role="probe_train")
        elif fault == "standardization":
            policy = replace(
                state.policy,
                observation_standardization=ObservationStandardization((1.0,) * 12, (2.0,) * 12, 2),
            )
        else:
            policy = replace(state.policy, binary_gripper_axis=3)
        source._policy, source._state = policy, replace(state, policy=policy)
        monkeypatch.setattr(source, "fork", forbidden)
    monkeypatch.setattr(continuation, "ActorShadowRuntime", forbidden)
    with pytest.raises(ValueError):
        continuation.continue_training(source, model, updates=1)


def controlled_zero_source():
    """Literal valid native state for metadata guards, without __init__ or RNG."""
    base = episode(20)
    rows = base.transitions[:2]
    raw = replace(
        base,
        episode_id="pick-cube-observed-grasp-teacher-20",
        context="pick-cube-train-20",
        transitions=(rows[0], replace(rows[1], terminated=True)),
    )
    model = history.action_history_training_episode(raw, PickCubeTaskSpaceHistoryRecipe())
    policy = bp.ContinuousBCPolicy(model.spec, hidden_dim=2, seed=17)
    parameters = (
        np.array([[0.0, 0.0]] * 12, dtype=np.float64),
        np.array([0.0, 0.0], dtype=np.float64),
        np.array([[0.0] * 4] * 2, dtype=np.float64),
        np.array([0.0] * 4, dtype=np.float64),
    )
    source = object.__new__(bp.NumpyContinuousBCLearner)
    source._policy = policy
    source._state = bp.ContinuousBCSnapshot(
        policy,
        parameters,
        (parameters[0].copy(), parameters[1].copy(), parameters[2].copy(), parameters[3].copy()),
        0,
        0,
    )
    return source, model


@pytest.mark.parametrize(
    "fault", ["budget_birth", "candidate_state", "candidate_policy", "event_scalar"]
)
def test_should_refuse_metadata_corruption_before_native_update(continuation, monkeypatch, fault):
    # This case deliberately does not request inputs: a metadata-only selection
    # may run separately without consuming another initializer or native update.
    source, model = controlled_zero_source()
    runtime_factory, snapshot, encode_json = (
        continuation.ActorShadowRuntime,
        source.snapshot_state,
        continuation._json,
    )
    captured: dict[str, Any] = dict(changed=False)

    def runtime(*args, **options):
        owner = runtime_factory(*args, **options)
        captured.update(runtime=owner, budget=options["budget"])
        if fault == "budget_birth":
            options["budget"].started_at -= 1.0
            captured["changed"] = True
        return owner

    def minimal_json(value):
        event_keys = {
            "index",
            "observed_at",
            "arrived_at",
            "deferred_exact",
            "update_attempted",
            "success_validated",
        }
        if type(value) is dict and set(value) == event_keys and "event" not in captured:
            captured["event"] = value
        return encode_json(value)

    def observe_source():
        returned = snapshot()
        if "runtime" in captured and not captured["changed"]:
            candidate = captured["runtime"]._candidate._learner
            if fault == "candidate_state":
                captured["original"] = candidate._state
                candidate._state = replace(candidate._state)
                captured["changed"] = True
            elif fault == "candidate_policy":
                captured["original"] = candidate._policy
                candidate._policy = replace(candidate._policy)
                captured["changed"] = True
            elif fault == "event_scalar" and "event" in captured:
                captured["event"]["index"] = 1
                captured["changed"] = True
        return returned

    monkeypatch.setattr(continuation, "ActorShadowRuntime", runtime)
    monkeypatch.setattr(continuation, "_json", minimal_json)
    monkeypatch.setattr(source, "snapshot_state", observe_source)
    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "__init__", forbidden)
    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "train_head_gauss_newton_batch", forbidden)
    result = continuation.continue_training(source, model, updates=1)
    report = result.report
    assert captured["changed"] is True
    assert report["complete"] is False and result.candidate is None
    assert (
        report["native_updates"]
        == report["sealed_native_updates"]
        == report["validated_native_updates"]
        == []
    )
    assert report["applied_updates"] == report["sealed_applied_updates"] == []
    assert captured["budget"].updates_completed == report["progress"]["updates_completed"] == 0
    assert report["update_attempts"] == report["attempted_training_rows"] == 0
    if fault in ("candidate_state", "candidate_policy"):
        candidate = captured["runtime"]._candidate._learner
        current = candidate._state if fault == "candidate_state" else candidate._policy
        assert current is not captured["original"]
    elif fault == "event_scalar":
        assert report["events"][0]["index"] == 1
