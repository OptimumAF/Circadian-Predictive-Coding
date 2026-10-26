"""H32 fixed old-A admission, ownership and paid failures without native work.

Native models may be constructed/copied. Every actual native update, prediction
and physics port is forbidden. Failure tests inject a raising adapter port before
its native implementation; no successful learning is inferred from fixtures.
"""

from copy import deepcopy
from dataclasses import replace
import pickle
import random

import numpy as np
import pytest

from src.adapters import minigrid_fixed_rehearsal as adapter
from src.adapters import minigrid_learnability as pilot
from src.adapters import minigrid_probe_rehearsal as recipe
from src.adapters.invariant_torch_feature_learner import (
    InvariantTorchFeatureLearner,
    InvariantTorchFeaturePolicy,
)
from src.adapters.torch_feature_learner import TorchFeatureLearner, TorchFeaturePolicy
from src.adapters.torch_feature_rehearsal_pilot import _same_state
from src.app.actor_shadow import ActorShadowRuntime
from src.app import toy_execution_budget as budgets
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.experience import LogicalClock
from src.shared.torch_runtime import require_torch
from test_minigrid_distinct_probe import raw_probe_data
from test_minigrid_invariant_learnability import train_sources
from test_minigrid_joint_catalogue_route import METHODS
from test_torch_feature_rehearsal_pilot import Sampler


def forbidden(*args, **kwargs):
    raise AssertionError("H32 rehearsal fixtures admit no native update, prediction or physics")


@pytest.fixture(autouse=True)
def no_native_work(monkeypatch):
    for owner in (InvariantTorchFeatureLearner, TorchFeatureLearner):
        monkeypatch.setattr(owner, "train_batch", forbidden)
        monkeypatch.setattr(owner, "predict", forbidden)
    monkeypatch.setattr(pilot, "MiniGridKeyDoor", forbidden)
    monkeypatch.setattr(pilot, "CueKeyDoorTeacher", forbidden)


@pytest.fixture(scope="module")
def native_sources():
    torch = require_torch()
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    return {
        method: InvariantTorchFeatureLearner(
            InvariantTorchFeaturePolicy(TorchFeaturePolicy(method, 296, 32, 17, num_classes=7))
        )
        for method in METHODS
    }


def sessions():
    clock = lambda: 1.0
    return (
        ToyBudgetSession(ToyExecutionBudget(max_training_updates=16), clock),
        ToyBudgetSession(ToyExecutionBudget(max_replay_examples=288), clock),
    )


def bank():
    return tuple(packet.episode for packet in train_sources()[:2])


def build(source, wake, replay, **options):
    return adapter.build_minigrid_old_a_rehearsal(
        source,
        wake,
        bank(),
        replay_budget=replay,
        total_updates=16,
        wide_neural=True,
        **options,
    )


@pytest.mark.parametrize("method", METHODS)
def test_should_admit_original_h32_without_learning_or_renewing_budgets(native_sources, method):
    source = native_sources[method]
    initial = source.snapshot_state()
    torch = require_torch()
    rng = torch.get_rng_state().clone(), pickle.dumps(np.random.get_state()), random.getstate()
    wake, replay = sessions()
    authority = (
        wake.budget,
        replay.budget,
        wake.clock,
        replay.clock,
        wake.started_at,
        replay.started_at,
    )
    plan = build(source, wake, replay, wake_role="probe_train")
    helper = plan.rehearsal
    assert plan.validation_projection_rows == 36
    assert plan.validation_projected_array_equivalent_bytes == 85248
    assert helper.retention.example_count == 18 and helper.retention.retained_bytes == 87280
    report = helper.report
    assert report["control"] == "fixed_train_episodes_v1"
    assert report["wake_role"] == "probe_train"
    assert report["fixed_bank"]["episode_ids"] == [episode.episode_id for episode in bank()]
    assert report["fixed_bank"]["construction_updates"] == 0
    assert report["planned_events"] == 16
    assert report["attempted_updates"] == report["returned_updates"] == 0
    assert helper.bind(source, wake, 16, 296, wake_role="probe_train") == 16
    assert wake.updates_completed == replay.replay_examples_completed == 0
    assert authority == (
        wake.budget,
        replay.budget,
        wake.clock,
        replay.clock,
        wake.started_at,
        replay.started_at,
    )
    assert _same_state(source.snapshot_state(), initial)
    assert torch.equal(torch.get_rng_state(), rng[0])
    assert pickle.dumps(np.random.get_state()) == rng[1] and random.getstate() == rng[2]
    report["fixed_bank"]["episode_ids"].clear()
    assert len(helper.report["fixed_bank"]["episode_ids"]) == 2


def test_should_preserve_h8_default_builder_behavior_exactly():
    source = InvariantTorchFeatureLearner(
        InvariantTorchFeaturePolicy(TorchFeaturePolicy("bp", 296, 8, 17, num_classes=7))
    )
    first_wake, first_replay = sessions()
    second_wake, second_replay = sessions()
    original = adapter.build_minigrid_old_a_rehearsal(
        source, first_wake, bank(), replay_budget=first_replay, total_updates=16
    )
    explicit = adapter.build_minigrid_old_a_rehearsal(
        source,
        second_wake,
        bank(),
        replay_budget=second_replay,
        total_updates=16,
        wide_neural=False,
    )
    assert original.rehearsal.report == explicit.rehearsal.report
    assert original.validation_projection_rows == explicit.validation_projection_rows
    assert (
        original.validation_projected_array_equivalent_bytes
        == explicit.validation_projected_array_equivalent_bytes
    )


@pytest.mark.parametrize("value", [None, 0, 1, "true", np.bool_(True)])
def test_should_refuse_non_native_explicit_width_before_charges(native_sources, value):
    wake, replay = sessions()
    with pytest.raises(ValueError):
        adapter.build_minigrid_old_a_rehearsal(
            native_sources["bp"],
            wake,
            bank(),
            replay_budget=replay,
            total_updates=16,
            wide_neural=value,
        )
    assert wake.updates_completed == replay.replay_examples_completed == 0


@pytest.mark.parametrize("wide", [False, True])
def test_should_refuse_implicit_or_mismatched_width(native_sources, wide):
    source = (
        native_sources["bp"]
        if not wide
        else InvariantTorchFeatureLearner(
            InvariantTorchFeaturePolicy(TorchFeaturePolicy("bp", 296, 8, 17, num_classes=7))
        )
    )
    wake, replay = sessions()
    with pytest.raises(ValueError):
        adapter.build_minigrid_old_a_rehearsal(
            source, wake, bank(), replay_budget=replay, total_updates=16, wide_neural=wide
        )
    assert wake.updates_completed == replay.replay_examples_completed == 0


@pytest.mark.parametrize(
    "changed",
    [
        {"hidden_dim": 16},
        {"seed": 18},
        {"learning_rate": 0.04},
        {"momentum": 0.8},
        {"inference_steps": 3},
        {"inference_learning_rate": 0.05},
    ],
)
def test_should_refuse_other_h32_native_recipes_before_replay(changed):
    policy = replace(TorchFeaturePolicy("bp", 296, 32, 17, num_classes=7), **changed)
    source = InvariantTorchFeatureLearner(InvariantTorchFeaturePolicy(policy))
    wake, replay = sessions()
    with pytest.raises(ValueError):
        build(source, wake, replay)
    assert wake.updates_completed == replay.replay_examples_completed == 0


@pytest.mark.parametrize(
    "fault", ["final_test", "probe_train", "probe_test", "context", "duplicate", "list", "extra"]
)
def test_should_refuse_reserved_or_malformed_old_bank_before_any_native_admission(
    native_sources, fault
):
    selected = bank()
    if fault in ("final_test", "probe_train", "probe_test"):
        selected = (selected[0], replace(selected[1], role=fault))
    elif fault == "context":
        selected = (selected[0], replace(selected[1], context="F"))
    elif fault == "duplicate":
        selected = (selected[0], selected[0])
    elif fault == "list":
        selected = list(selected)
    else:
        object.__setattr__(selected[1], "foreign", True)
    wake, replay = sessions()
    with pytest.raises(ValueError):
        adapter.build_minigrid_old_a_rehearsal(
            native_sources["bp"],
            wake,
            selected,
            replay_budget=replay,
            total_updates=16,
            wide_neural=True,
        )
    assert wake.updates_completed == replay.replay_examples_completed == 0


@pytest.mark.parametrize("fault", ["contents", "arrays", "membership", "source", "wake"])
def test_should_refuse_replaced_bank_source_or_wake_before_binding(native_sources, fault):
    source = native_sources["bp"]
    wake, replay = sessions()
    helper = build(source, wake, replay).rehearsal
    key = next(iter(helper._buffer._rows))
    features, targets = helper._buffer._rows[key]
    if fault == "contents":
        features[0, -3] += 0.125
    elif fault == "arrays":
        helper._buffer._rows[key] = (features.copy(), targets.copy())
    elif fault == "membership":
        helper._buffer._rows["replacement"] = helper._buffer._rows.pop(key)
    elif fault == "source":
        source = source.fork()
    else:
        wake, _ = sessions()
    with pytest.raises(ValueError):
        helper.bind(source, wake, 16, 296)
    assert wake.updates_completed == replay.replay_examples_completed == 0


@pytest.mark.parametrize("method", METHODS)
def test_should_keep_paid_replay_attempt_after_adapter_failure_without_native_update(
    monkeypatch, native_sources, method
):
    source = native_sources[method]
    initial = source.snapshot_state()
    wake, replay = sessions()
    helper = build(source, wake, replay).rehearsal
    runtime = ActorShadowRuntime(
        source,
        actor_version="original",
        candidate_version="disposable",
        clock=LogicalClock(),
        budget=wake,
    )
    candidate = runtime._candidate.snapshot_state()
    attempts = []

    def fail_before_native(owner, features, targets):
        attempts.append((features.shape, targets.shape, replay.replay_examples_completed))
        raise RuntimeError("injected adapter failure before native implementation")

    monkeypatch.setattr(InvariantTorchFeatureLearner, "train_batch", fail_before_native)
    with pytest.raises(RuntimeError, match="injected adapter failure"):
        helper._run_event(runtime, helper._events[0])
    report = helper.report
    assert attempts == [((9, 296), (9, 1), 9)]
    assert report["attempted_updates"] == 1 and report["charged_exposures"] == 9
    assert report["returned_updates"] == report["committed_updates"] == 0
    assert report["events"][0]["status"] == "failed"
    assert replay.replay_examples_completed == 9 and wake.updates_completed == 0
    assert runtime.applied_consolidations == ()
    assert _same_state(runtime._candidate.snapshot_state(), candidate)
    assert _same_state(source.snapshot_state(), initial)
    paid = deepcopy(helper._events[0])
    next(iter(helper._buffer._rows.values()))[0][0, -3] += 0.125
    with pytest.raises(ValueError, match="fixed training bank content changed"):
        helper._run_event(runtime, helper._events[1])
    assert helper._events[0] == paid
    assert replay.replay_examples_completed == 9 and len(attempts) == 1


@pytest.mark.parametrize("method", METHODS)
def test_should_compose_wide_probe_replay_with_original_roles_and_failure_accounting(
    monkeypatch, native_sources, method
):
    source = native_sources[method]
    state = source.snapshot_state()
    data = tuple(pilot._history_packets(packets) for packets in raw_probe_data())
    monitors = train_sources()[:2]
    original_inputs = pickle.dumps((data, monitors))
    benchmark = ToyBudgetSession(ToyExecutionBudget(max_training_updates=0), lambda: 1.0)
    binding = pilot._budget_binding(benchmark)
    row = pilot._new_row(f"public_invariant_{method}")
    slot = row["probes"][0]
    plans = []
    builder = pilot.build_minigrid_old_a_rehearsal

    def capture(*args, **options):
        assert options["wide_neural"] is True and options["wake_role"] == "probe_train"
        assert options["total_updates"] == 16
        assert all(episode.role == "train" and episode.context == "A" for episode in args[2])
        plan = builder(*args, **options)
        plans.append((plan, options["replay_budget"]))
        return plan

    def stop_after_factory(owner, checkpoint, training, evaluation, **options):
        assert owner is source and _same_state(checkpoint, state)
        assert all(packet.role == "probe_train" for packet in training)
        assert all(packet.role == "probe_test" for packet in evaluation)
        assert all(packet.role == "train" for packet in options["training_readouts"])
        assert options["budget"] is not options["return_budget"]
        helper = options["rehearsal_factory"](owner, options["budget"], 16)
        assert helper.report["wake_role"] == "probe_train"
        raise RuntimeError("scripted stop after H32 rehearsal admission")

    monkeypatch.setattr(pilot, "monotonic", lambda: 1.0)
    monkeypatch.setattr(budgets, "ProcessRssSampler", Sampler)
    monkeypatch.setattr(pilot, "build_minigrid_old_a_rehearsal", capture)
    monkeypatch.setattr(pilot, "run_fresh_task_probe", stop_after_factory)
    with pytest.raises(RuntimeError, match="scripted stop after H32 rehearsal admission"):
        pilot._run_probe(
            slot,
            dict(source=source, row=row, budget=benchmark),
            state,
            data,
            sampler=Sampler(),
            deadline=10.0,
            check_boundary=lambda: None,
            training_readouts=monitors,
            return_to_a=True,
            catalogue_rows=54,
            old_a_rehearsal=True,
            wide_neural=True,
        )
    assert len(plans) == 1
    assert slot["status"] == "failed" and slot["rehearsal_work"]["work_unknown"] is True
    assert slot["rehearsal_work"]["bank_encoded_bytes"] == 87280
    assert slot["rehearsal_work"]["charged_replay_examples"] == 0
    assert plans[0][1].replay_examples_completed == 0
    assert benchmark.updates_completed == 0 and pilot._budget_binding(benchmark) == binding
    assert _same_state(source.snapshot_state(), state)
    assert pickle.dumps((data, monitors)) == original_inputs
    aggregated = recipe.aggregate([row])
    assert aggregated["partial_work_unknown"] is True
    assert aggregated["logical_bank_encoded_bytes"] == 87280
