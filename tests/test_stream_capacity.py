"""Capacity boundaries on real runtime receipts, with tiny guarded native growth.

Fakes isolate observer failures. Two native trajectories measure the model used
for each wake, while one-shot refusal/fault controls retain charged work.
"""

from copy import deepcopy
from dataclasses import asdict
from typing import Any

import numpy as np
import pytest

from src.adapters.coupled_learners import CoupledLearner
from src.adapters.numpy_capacity import measure_numpy_capacity
from src.adapters.structural_coupled_scaling_learner import StructuralCoupledScalingLearner
import src.app.stream_capacity as capacity_module
from src.app.actor_shadow import ActorShadowRuntime
from src.app.stream_capacity import StreamCapacity
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.capacity_cost import CapacityUsage
from src.core.coupled_circadian_predictive_coding import CoupledCircadianConfig
from src.core.coupled_pc_scaling import CoupledScalingConfig, initialize_coupled_scaling
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.learner_ports import TrainingDiagnostic


class FakeLearner:
    def __init__(self, state=None):
        self.state = {"updates": 0} if state is None else deepcopy(state)

    def fork(self):
        return FakeLearner(self.state)

    def snapshot_state(self):
        return deepcopy(self.state)

    def restore_state(self, state):
        self.state = deepcopy(state)

    def train_batch(self, features, targets):
        self.state["updates"] += 1
        return TrainingDiagnostic("capacity_fixture_count", float(self.state["updates"]))


def scalar_measure(state):
    return CapacityUsage(13, 104, 0)


def make_helper(learner=None, *, total=4, measure=scalar_measure, seconds=None, **changes):
    learner = FakeLearner() if learner is None else learner
    seconds = [0.0] if seconds is None else seconds
    budget = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=total, max_wall_seconds=60), lambda: seconds[0]
    )
    options: dict[str, Any] = dict(
        total_updates=total, measure_state=measure, max_parameters=17, max_state_array_bytes=200
    )
    options.update(changes)
    helper = StreamCapacity(learner, budget, **options)
    return helper, learner, budget, seconds


def runtime_for(helper, learner, budget, total=4):
    limit = helper.bind(learner, budget, total)
    clock = LogicalClock()
    runtime = ActorShadowRuntime(
        learner,
        actor_version="a0",
        candidate_version="c0",
        clock=clock,
        budget=budget,
        max_experiences=total,
        max_consolidations=limit,
    )
    helper.start(runtime)
    return runtime, clock


def wake(runtime, clock, helper, index, rows=1):
    clock.advance_to(index)
    runtime.record_experience(
        Experience(
            f"sample-{index}",
            "capacity",
            index,
            "a0",
            np.tile(np.array([[0.2, -0.4]]), (rows, 1)),
            "train",
            ExperiencePermissions(training=True),
        )
    )
    runtime.record_label(
        LabelArrival(
            f"label-{index}", f"sample-{index}", "capacity", index, "a0", np.ones((rows, 1))
        )
    )
    assert len(runtime.train_ready(max_updates=1)) == 1
    helper.after_update(runtime, index, rows)


def structural():
    initial = initialize_coupled_scaling(
        CoupledScalingConfig("mean_field", 2, 2, 1, 1.5, (1.0,), CoupledCircadianConfig()),
        seed=23,
    )
    return StructuralCoupledScalingLearner(
        initial, learning_rate=0.01, inference_steps=2, inference_learning_rate=0.05
    )


@pytest.mark.parametrize(
    "change",
    [
        {"total_updates": True},
        {"total_updates": 0},
        {"measure_state": None},
        {"split_at_update": True, "max_width": 4},
    ],
)
def test_should_refuse_invalid_capacity_configuration_without_native_work(change):
    with pytest.raises(ValueError):
        make_helper(**change)


@pytest.mark.parametrize("split", [False, True])
def test_should_weight_each_successful_wake_by_its_pre_wake_committed_capacity(split):
    learner = (
        structural()
        if split
        else CoupledLearner(
            CoupledPredictiveCodingNetwork(2, (4,), seed=23),
            learning_rate=0.01,
            inference_steps=2,
            inference_learning_rate=0.05,
        )
    )
    before = learner.snapshot_state()
    helper, _, budget, _ = make_helper(
        learner,
        measure=measure_numpy_capacity,
        split_at_update=2 if split else None,
        max_width=4 if split else None,
    )
    runtime, clock = runtime_for(helper, learner, budget)
    for index in range(1, 5):
        wake(runtime, clock, helper, index, rows=2 if split and index > 2 else 1)
    report = helper.report
    # Independent dense counts: width2 has9 parameters, width4 has17.
    assert report["completed_native_calls"] == 4
    assert report["completed_training_rows"] == (6 if split else 4)
    assert report["peak_parameters"] == 17
    assert report["mean_parameters"] == (13 if split else 17)
    assert report["mean_parameter_bytes"] == (104 if split else 136)
    assert report["mean_total_array_bytes"] == (152 if split else 136)
    assert report["mean_auxiliary_bytes"] == (48 if split else 0)
    assert report["peak_total_array_bytes"] == (200 if split else 136)
    assert report["peak_auxiliary_bytes"] == (64 if split else 0)
    assert len(report["samples"]) == (6 if split else 5)
    assert budget.updates_completed == 4
    assert len(runtime.applied_consolidations) == int(split)
    np.testing.assert_equal(asdict(learner.snapshot_state()), asdict(before))


@pytest.mark.parametrize("limit", ["parameters", "bytes", "width"])
def test_should_refuse_expansion_before_the_native_transform_when_any_target_cap_fails(
    monkeypatch, limit
):
    changes = (
        {"max_parameters": 16}
        if limit == "parameters"
        else ({"max_state_array_bytes": 199} if limit == "bytes" else {"max_width": 3})
    )
    options = {"split_at_update": 1, "max_width": 4, **changes}
    helper, learner, budget, _ = make_helper(
        structural(), measure=measure_numpy_capacity, **options
    )
    runtime, clock = runtime_for(helper, learner, budget)

    def forbidden_transform(*args, **kwargs):
        raise AssertionError("predicted target caps must precede native transform")

    monkeypatch.setattr(
        capacity_module.CoupledScalingSplitConsolidator, "__call__", forbidden_transform
    )
    with pytest.raises((ValueError, RuntimeError)):
        wake(runtime, clock, helper, 1)
    assert budget.updates_completed == 1 and len(runtime.applied_updates) == 1
    assert not runtime.applied_consolidations
    report = helper.report
    assert report["completed_native_calls"] == 1
    assert report["split_requested"] == 1
    assert report["split_attempted"] == report["split_committed"] == 0
    event = report["events"][0]
    assert event["requested"] and not event["attempted"] and not event["committed"]


@pytest.mark.parametrize("fault", ["raise", "wrong_record", "growth"])
def test_should_preserve_completed_wake_work_when_postwake_measurement_fails(fault):
    def measure(state):
        if state["updates"]:
            if fault == "raise":
                raise RuntimeError("measurement unavailable")
            if fault == "wrong_record":
                return {"parameters": 13}
            return CapacityUsage(17, 136, 0)
        return CapacityUsage(13, 104, 0)

    helper, learner, budget, _ = make_helper(measure=measure)
    runtime, clock = runtime_for(helper, learner, budget)
    with pytest.raises((ValueError, RuntimeError)):
        wake(runtime, clock, helper, 1)
    assert budget.updates_completed == 1 and len(runtime.applied_updates) == 1
    assert helper.report["completed_native_calls"] == 1
    assert helper.report["completed_training_rows"] == 1


@pytest.mark.parametrize("operation", ["reuse", "widen_ceiling"])
def test_should_bind_only_the_original_fresh_owner_once_without_widening_birth_limits(operation):
    helper, learner, budget, _ = make_helper()
    if operation == "widen_ceiling":
        helper._max_parameters = 18
        with pytest.raises(ValueError):
            helper.bind(learner, budget, 4)
        assert budget.updates_completed == 0
        assert helper.report["completed_native_calls"] == 0
        return
    with pytest.raises(ValueError):
        helper.bind(learner.fork(), budget, 4)
    assert helper.bind(learner, budget, 4) == 0
    with pytest.raises(ValueError):
        helper.bind(learner, budget, 4)


def test_should_require_actual_wake_receipts_before_counting_work():
    helper, learner, budget, _ = make_helper()
    runtime, _ = runtime_for(helper, learner, budget)
    with pytest.raises(ValueError):
        helper.after_update(runtime, 1, 1)
    assert budget.updates_completed == 0
    assert helper.report["completed_native_calls"] == 0


@pytest.mark.parametrize("fault", ["transform", "after_publication"])
def test_should_report_failed_split_work_truthfully_without_refunding_or_retrying(
    monkeypatch, fault
):
    helper, learner, budget, _ = make_helper(
        structural(), measure=measure_numpy_capacity, split_at_update=1, max_width=4
    )
    runtime, clock = runtime_for(helper, learner, budget)
    if fault == "transform":

        def fail(*args, **kwargs):
            raise RuntimeError("transform unavailable")

        monkeypatch.setattr(capacity_module.CoupledScalingSplitConsolidator, "__call__", fail)
    else:
        consolidate = runtime.consolidate

        def fail(*args, **kwargs):
            consolidate(*args, **kwargs)
            raise RuntimeError("after committed publication")

        monkeypatch.setattr(runtime, "consolidate", fail)
    with pytest.raises(RuntimeError):
        wake(runtime, clock, helper, 1)
    assert budget.updates_completed == 1
    assert len(runtime.applied_consolidations) == int(fault == "after_publication")
    report = helper.report
    assert report["completed_native_calls"] == 1
    assert report["peak_parameters"] == (17 if fault == "after_publication" else 9)
    assert report["split_requested"] == report["split_attempted"] == 1
    assert report["split_committed"] == int(fault == "after_publication")
    assert (report["events"][0]["receipt"] is not None) == (fault == "after_publication")
    with pytest.raises(ValueError):
        helper.after_update(runtime, 1, 1)


def test_should_return_detached_bounded_scalar_reports_without_retaining_native_snapshots():
    helper, learner, budget, _ = make_helper()
    runtime, clock = runtime_for(helper, learner, budget)
    for index in range(1, 5):
        wake(runtime, clock, helper, index)
    before = deepcopy(helper.report)
    changed = helper.report
    changed.clear()
    assert helper.report == before
    assert len(before["samples"]) == 5
    assert before["mean_parameters"] == 13
