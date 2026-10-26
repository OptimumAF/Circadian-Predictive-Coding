"""A spent guarded prune schedule at actual committed-capacity boundaries.

Known zero-damage tensors establish acceptance; an independently nonzero removed
path establishes rejection. Neither fixture selects the comparison's thresholds.
"""

from copy import deepcopy
from dataclasses import asdict
from typing import Any

import numpy as np
import pytest

from src.adapters.numpy_capacity import measure_numpy_capacity
from src.adapters.structural_coupled_scaling_learner import StructuralCoupledScalingLearner
from src.adapters.structural_scaled_matched_backprop_learner import (
    StructuralScaledMatchedBackpropLearner,
)
from src.app.actor_shadow import ActorShadowRuntime
from src.app.coupled_pruning_gate import CoupledPruneLimits
from src.app.linear_scaling_measurement import ScalingBatch
from src.app.stream_capacity import StreamCapacity
import src.app.stream_capacity_prune as prune_module
from src.app.stream_capacity_prune import StreamCapacityPrune
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.coupled_pc_scaling import (
    CoupledScalingConfig,
    initialize_coupled_scaling,
    step_coupled_scaling,
)
from src.core.coupled_structural_state import CoupledStructuralState
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.learner_ports import TrainingDiagnostic
from src.core.scaled_matched_backprop import step_scaled_matched_backprop


def raw(depth=1, *, damaging=False):
    state = initialize_coupled_scaling(
        CoupledScalingConfig("mean_field", 2, 2, depth, 1.5), seed=23
    )
    p = state.raw_parameters
    for value in (*p.hidden_weights, *p.hidden_biases, p.output_weight, p.output_bias):
        value.fill(0)
    if damaging:
        assert depth == 1
        p.hidden_weights[0][0, 1] = 1.0
        p.output_weight[1, 0] = 1.0
    return state


def probes():
    return (
        ScalingBatch(
            "validation", ("reserved-validation",), np.array([[1.0, 0.0]]), np.array([[0.5]])
        ),
    )


def budget(total=3, seconds=None):
    seconds = [0.0] if seconds is None else seconds
    return ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=total, max_wall_seconds=60), lambda: seconds[0]
    )


def driver(session, *, at=2, removed=(2,), supplied=None, limits=None, **changes):
    options: dict[str, Any] = dict(
        at_update=at,
        removed_units=removed,
        probes=probes() if supplied is None else supplied,
        limits=CoupledPruneLimits(0.0, 0.0, 0.0) if limits is None else limits,
        budget=session,
        inference_steps=2,
        inference_learning_rate=0.05,
    )
    options.update(changes)
    return StreamCapacityPrune(**options)


def prepare(source, session, pruning, *, total=3, split=None, measure=measure_numpy_capacity):
    cap = StreamCapacity(
        source,
        session,
        total_updates=total,
        measure_state=measure,
        max_parameters=40,
        max_state_array_bytes=512,
        split_at_update=split,
        max_width=4 if split is not None else None,
        pruning=pruning,
    )
    maximum = cap.bind(source, session, total)
    clock = LogicalClock()
    runtime = ActorShadowRuntime(
        source,
        actor_version="a0",
        candidate_version="c0",
        clock=clock,
        budget=session,
        max_experiences=total,
        max_consolidations=maximum,
    )
    cap.start(runtime)
    return cap, runtime, clock


def wake(cap, runtime, clock, number):
    clock.advance_to(number)
    runtime.record_experience(
        Experience(
            f"sample-{number}",
            "prune-capacity",
            number,
            "a0",
            np.zeros((1, 2)),
            "train",
            ExperiencePermissions(training=True),
        )
    )
    runtime.record_label(
        LabelArrival(
            f"label-{number}",
            f"sample-{number}",
            "prune-capacity",
            number,
            "a0",
            np.full((1, 1), 0.5),
        )
    )
    assert len(runtime.train_ready(max_updates=1)) == 1
    cap.after_update(runtime, number, 1)


def same(actual, expected):
    np.testing.assert_equal(asdict(actual), asdict(expected))


@pytest.mark.parametrize("method", ["pc", "bp"])
@pytest.mark.parametrize("depth", [1, 2])
def test_should_split_then_accept_zero_damage_pruning_and_continue_the_exact_raw_learner(
    method, depth
):
    initial = raw(depth)
    source = (
        StructuralScaledMatchedBackpropLearner(initial, learning_rate=0.01)
        if method == "bp"
        else StructuralCoupledScalingLearner(
            initial, learning_rate=0.01, inference_steps=2, inference_learning_rate=0.05
        )
    )
    source_before = source.snapshot_state()
    session = budget()
    pruning = driver(session, removed=(2,) * depth)
    cap, runtime, clock = prepare(source, session, pruning, split=1)
    wake(cap, runtime, clock, 1)
    assert runtime.candidate_snapshot().state.raw_state.config.width == 4
    wake(cap, runtime, clock, 2)
    pruned = runtime.candidate_snapshot().state
    assert pruned.raw_state.config.width == 3
    assert [item.kind for item in pruned.transitions] == ["split", "prune"]
    assert len(runtime.applied_consolidations) == 2
    wake(cap, runtime, clock, 3)
    x, y = np.zeros((1, 2)), np.full((1, 1), 0.5)
    reference = (
        step_scaled_matched_backprop(pruned.raw_state, x, y, 0.01)
        if method == "bp"
        else step_coupled_scaling(pruned.raw_state, x, y, 0.01, 2, 0.05)
    )
    same(runtime.candidate_snapshot().state.raw_state, reference.state)
    same(source.snapshot_state(), source_before)
    report = cap.report
    # Independent D2 dense weights+biases at widths2/4/3, without calling
    # the production counting or measurement function for the expected mean.
    counts = (9, 17, 13) if depth == 1 else (15, 37, 25)
    assert report["completed_native_calls"] == report["completed_training_rows"] == 3
    assert report["mean_parameters"] == sum(counts) / 3
    assert report["peak_parameters"] == counts[1]
    assert report["current_usage"]["parameters"] == counts[2]
    assert [sample["phase"] for sample in report["samples"]].count("post_prune") == 1
    assert cap.observation_only is False
    event = report["pruning"]["events"][0]
    assert event["decision_status"] == "accepted" and event["committed"]
    assert event["known_validation_score_calls"] == 2
    assert event["known_validation_row_forwards"] == 2
    assert event["known_validation_energy_evaluations"] == 6
    assert event["validation_work_complete"] is True
    assert len(event["diagnostics"]) == 1


def test_should_record_actual_damage_rejection_without_publication_and_continue_training():
    initial = raw(damaging=True)
    source = StructuralCoupledScalingLearner(
        initial, learning_rate=0.01, inference_steps=2, inference_learning_rate=0.05
    )
    session = budget(2)
    pruning = driver(session, at=1, removed=(1,))
    cap, runtime, clock = prepare(source, session, pruning, total=2)
    wake(cap, runtime, clock, 1)
    state = runtime.candidate_snapshot().state
    assert state.raw_state.config.width == 2 and state.transitions == ()
    assert not runtime.applied_consolidations
    before_second = deepcopy(state)
    wake(cap, runtime, clock, 2)
    same(runtime.candidate_snapshot().state, before_second)
    assert session.updates_completed == cap.report["completed_native_calls"] == 2
    assert not cap.report["stopped"]
    event = pruning.report["events"][0]
    assert event["decision_status"] == event["status"] == "rejected"
    assert event["attempted"] and not event["committed"] and event["receipt"] is None
    assert event["diagnostics"][0]["max_prediction_change"] > 0


class FakeStructural:
    def __init__(self, state=None, trace=None):
        if state is None:
            initial = raw()
            self.state = CoupledStructuralState(1, initial.config, initial)
        else:
            self.state = deepcopy(state)
        self.trace: list[str] = [] if trace is None else trace

    def fork(self):
        return FakeStructural(self.state, self.trace)

    def snapshot_state(self):
        return deepcopy(self.state)

    def restore_state(self, state):
        self.state = deepcopy(state)

    def train_batch(self, features, targets):
        self.trace.append("fake_train")
        return TrainingDiagnostic("explicit_fake_noop", 0.0)


@pytest.mark.parametrize(
    "change",
    [
        {"at_update": 0},
        {"at_update": True},
        {"removed_units": (True,)},
        {"inference_steps": 0},
    ],
)
def test_should_refuse_invalid_pruning_policy_before_any_transform(change):
    with pytest.raises(ValueError):
        driver(budget(), **change)


@pytest.mark.parametrize("role", ["final_test", "unknown"])
def test_should_refuse_nontraining_roles_before_copying_or_converting_probe_payloads(role):
    calls: list[str] = []

    class Poison:
        def __deepcopy__(self, memo):
            calls.append("copy")
            raise AssertionError("heldout payload must not be touched")

        def __array__(self, *args, **kwargs):
            calls.append("convert")
            raise AssertionError("heldout payload must not be touched")

    opaque: Any = Poison()
    supplied = (ScalingBatch(role, ("heldout",), opaque, opaque),)
    with pytest.raises(ValueError):
        driver(budget(), supplied=supplied)
    assert calls == []


@pytest.mark.parametrize("fault", ["schedule", "removed", "limits", "probes", "clock", "counter"])
def test_should_refuse_mutated_birth_policy_data_or_budget_before_binding_more_work(fault):
    session = budget()
    pruning = driver(session)
    if fault == "schedule":
        object.__setattr__(pruning._policy, "at_update", 1)
    elif fault == "removed":
        object.__setattr__(pruning._policy, "removed_units", (1,))
    elif fault == "limits":
        object.__setattr__(pruning._policy.limits, "max_prediction_change", 0.1)
    elif fault == "probes":
        pruning._probes[0].features[0, 0] = 2.0
    elif fault == "clock":
        session.clock = lambda: 0.0
    else:
        session.updates_completed = 1
    with pytest.raises(ValueError):
        pruning.bind(session, 3)
    assert not pruning.report["bound"]


@pytest.mark.parametrize(
    "fault", ["gate_error", "failed_gate", "late_commit", "secondary_measure", "expired_commit"]
)
def test_should_retain_primary_failures_spent_schedule_and_actual_published_work(
    monkeypatch, fault
):
    import src.app.coupled_scaling_pruning_gate as native_gate
    from src.app.toy_execution_budget import ToyExecutionStopped

    source, seconds = FakeStructural(), [0.0]
    session = budget(1, seconds)
    pruning = driver(session, at=1, removed=(1,))
    primary = RuntimeError("primary prune failure")
    consolidator = prune_module.CoupledScalingStructuralPruneConsolidator
    consolidate = ActorShadowRuntime.consolidate
    restore = FakeStructural.restore_state

    def post_commit(self, *args, **kwargs):
        consolidate(self, *args, **kwargs)
        raise primary

    def expire(self, state):
        restore(self, state)
        if state.raw_state.config.width == 1:
            seconds[0] = 60.0

    def measure(state):
        if fault == "secondary_measure" and state.raw_state.config.width == 1:
            raise LookupError("secondary read failure")
        return measure_numpy_capacity(state)

    if fault == "gate_error":

        def fail(*args, **kwargs):
            raise primary

        monkeypatch.setattr(consolidator, "__call__", fail)
    elif fault == "failed_gate":

        def fail(*args, **kwargs):
            raise RuntimeError("native validation failed")

        monkeypatch.setattr(native_gate, "_score", fail)
    elif fault == "expired_commit":
        monkeypatch.setattr(FakeStructural, "restore_state", expire)
    else:
        monkeypatch.setattr(ActorShadowRuntime, "consolidate", post_commit)
    cap, runtime, clock = prepare(source, session, pruning, total=1, measure=measure)
    expected_error = (
        ToyExecutionStopped
        if fault == "expired_commit"
        else (ValueError if fault == "failed_gate" else RuntimeError)
    )
    with pytest.raises(expected_error) as failure:
        wake(cap, runtime, clock, 1)
    if fault in ("gate_error", "late_commit", "secondary_measure"):
        assert failure.value is primary
    committed = fault in ("late_commit", "secondary_measure", "expired_commit")
    event = pruning.report["events"][0]
    assert event["attempted"] and event["committed"] is committed
    assert (event["receipt"] is not None) is committed
    assert len(runtime.applied_consolidations) == int(committed)
    assert session.updates_completed == cap.report["completed_native_calls"] == 1
    assert pruning.report["stopped"] and cap.report["stopped"]
    assert source.trace == ["fake_train"]
    if fault == "failed_gate":
        assert event["decision_status"] == "failed"
        assert event["known_validation_score_calls"] == 0
        assert event["validation_work_complete"] is False
        assert event["validation_score_calls"] is None
        assert event["validation_energy_evaluations"] is None
    if committed:
        assert event["decision_status"] == "accepted"
        assert cap.report["samples"][-1]["phase"] == "post_prune"
        if fault == "secondary_measure":
            assert event["measurement_error"] == "LookupError: secondary read failure"
        else:
            assert cap.report["current_usage"]["parameters"] == 5
    with pytest.raises(ValueError):
        cap.after_update(runtime, 1, 1)
    assert source.trace == ["fake_train"]


def test_should_own_caller_probe_limits_and_the_last_decision_without_an_extra_gate(monkeypatch):
    from src.app.coupled_structural_consolidation import CoupledScalingStructuralPruneConsolidator

    source, session = FakeStructural(), budget(1)
    supplied, limits = probes(), CoupledPruneLimits(0, 0, 0)
    pruning = driver(session, at=1, removed=(1,), supplied=supplied, limits=limits)
    default = CoupledScalingStructuralPruneConsolidator(
        (1,),
        probes(),
        limits=CoupledPruneLimits(0, 0, 0),
        budget=session,
        inference_steps=2,
        inference_learning_rate=0.05,
    )
    assert default.last_decision is None
    captured: list[Any] = []

    def construct(*args, **kwargs):
        instance = CoupledScalingStructuralPruneConsolidator(*args, **kwargs)
        captured.append(instance)
        return instance

    monkeypatch.setattr(prune_module, "CoupledScalingStructuralPruneConsolidator", construct)
    supplied[0].features.fill(np.nan)
    object.__setattr__(supplied[0], "role", "final_test")
    object.__setattr__(limits, "max_prediction_change", -1)
    cap, runtime, clock = prepare(source, session, pruning, total=1)
    wake(cap, runtime, clock, 1)
    assert len(captured) == 1
    decision = captured[0].last_decision
    assert decision is not None and decision.status == "accepted"
    decision.state.raw_parameters.output_bias.fill(999)
    again = captured[0].last_decision
    assert again is not None
    np.testing.assert_array_equal(again.state.raw_parameters.output_bias, np.zeros((1, 1)))
    detached = pruning.report
    detached["events"][0]["diagnostics"].clear()
    assert len(pruning.report["events"][0]["diagnostics"]) == 1
    assert default.last_decision is None


def test_should_keep_legacy_capacity_report_schema_when_pruning_is_absent():
    source, session = FakeStructural(), budget(1)
    cap = StreamCapacity(
        source,
        session,
        total_updates=1,
        measure_state=measure_numpy_capacity,
        max_parameters=17,
        max_state_array_bytes=200,
    )
    assert cap.observation_only and "pruning" not in cap.report
    assert cap.report["events"] == []


def test_should_refuse_pruning_with_renewal_before_binding_either_helper():
    from tests.test_renewal_capacity import FakeRenewable
    from src.app.coupled_attribution import AttributionBatch, run_attribution_stream
    from src.app.stream_unit_renewal import StreamUnitRenewal

    source, session = FakeRenewable(), budget(1)
    pruning = driver(session, at=1, removed=(1,))
    cap = StreamCapacity(
        source,
        session,
        total_updates=1,
        measure_state=measure_numpy_capacity,
        max_parameters=17,
        max_state_array_bytes=256,
        pruning=pruning,
    )
    renew = StreamUnitRenewal(source, session, every_updates=1, total_updates=1, max_replacements=1)
    training = (AttributionBatch("train", "A", ("train",), np.zeros((1, 2)), np.full((1, 1), 0.5)),)
    final = (AttributionBatch("final_test", "A", ("final",), np.ones((1, 2)), np.ones((1, 1))),)
    with pytest.raises(ValueError):
        run_attribution_stream(
            source,
            training,
            final,
            checkpoint_updates=(0, 1),
            label_delay=0,
            budget=session,
            renewal=renew,
            capacity=cap,
        )
    assert not cap.report["bound"] and not pruning.report["bound"] and not renew.report["bound"]
