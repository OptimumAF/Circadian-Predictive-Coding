"""Capacity observation around bounded renewal without changing native work.

Three tiny native pairs establish measurement noninterference. Explicit fake
learners isolate scheduling, authority and permanent-charge failure controls.
"""

from copy import deepcopy
from dataclasses import asdict, replace
import pickle
from typing import Any

import numpy as np
import pytest

from src.adapters.coupled_learners import CoupledLearner
from src.adapters.numpy_capacity import measure_numpy_capacity
from src.app.actor_shadow import ActorShadowRuntime
from src.app.coupled_attribution import AttributionBatch, AttributionStream, run_attribution_stream
from src.app.stream_capacity import StreamCapacity
from src.app.stream_rehearsal import StreamRehearsal, StreamReplayPolicy
from src.app.stream_unit_renewal import StreamUnitRenewal
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.capacity_cost import CapacityUsage
from src.core.circadian_predictive_coding import ReplayRetentionBudget
from src.core.coupled_circadian_predictive_coding import CoupledCircadianConfig
from src.core.coupled_energy import CoupledParameters
from src.core.coupled_unit_renewal import (
    RenewalPolicy,
    RenewalResult,
    advance_renewal_history,
    initial_renewal_history,
    select_renewal_indices,
)
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.learner_ports import TrainingDiagnostic
from src.core.renewable_coupled_predictive_coding import (
    RenewableCoupledCircadianPredictiveCodingNetwork,
    RenewableCoupledPredictiveCodingNetwork,
    RenewableCoupledSnapshot,
)
from src.core.shared_replay_schedule import ReplayAllocation


def data(total=4):
    training = tuple(
        AttributionBatch(
            "train",
            "A",
            (f"train-{k}-0", f"train-{k}-1"),
            np.zeros((2, 2)),
            np.full((2, 1), 0.5),
        )
        for k in range(total)
    )
    final = (AttributionBatch("final_test", "A", ("final-0",), np.ones((1, 2)), np.ones((1, 1))),)
    return training, final


def session(total=4):
    return ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=total, max_wall_seconds=60), lambda: 0.0
    )


def capacity(learner, budget, *, total=4, measure=measure_numpy_capacity, split=None):
    return StreamCapacity(
        learner,
        budget,
        total_updates=total,
        measure_state=measure,
        max_parameters=17,
        max_state_array_bytes=256,
        split_at_update=split,
        max_width=None if split is None else 4,
    )


def renewal(learner, budget, *, total=4, limit=1):
    return StreamUnitRenewal(
        learner, budget, every_updates=2, total_updates=total, max_replacements=limit
    )


def run(learner, budget, renew, cap=None, *, total=4, observer=None, **changes):
    training, final = data(total)
    return run_attribution_stream(
        learner,
        training,
        final,
        checkpoint_updates=(0, total),
        label_delay=1,
        budget=budget,
        renewal=renew,
        capacity=cap,
        observe_checkpoint=observer,
        **changes,
    )


def without_times(value):
    if type(value) is dict:
        return {
            key: without_times(item)
            for key, item in value.items()
            if key not in {"work_seconds", "event_seconds"}
        }
    if type(value) is list:
        return [without_times(item) for item in value]
    return value


def native(kind):
    options: dict[str, Any] = dict(renewal_policy=RenewalPolicy(2, 0.0, 1))
    if kind == "pc":
        model: Any = RenewableCoupledPredictiveCodingNetwork(2, (3,), seed=17, **options)
    else:
        options["config"] = (
            CoupledCircadianConfig.matched_pc_control()
            if kind == "neutral"
            else CoupledCircadianConfig()
        )
        model = RenewableCoupledCircadianPredictiveCodingNetwork(2, (3,), seed=17, **options)
    return CoupledLearner(
        model, learning_rate=0.02, inference_steps=2, inference_learning_rate=0.04
    )


@pytest.mark.parametrize("kind", ["pc", "neutral", "active"])
def test_should_measure_before_and_after_renewal_without_changing_any_native_trajectory(
    monkeypatch, kind
):
    reference, measured = native(kind), native(kind)
    before = pickle.dumps((reference.snapshot_state(), measured.snapshot_state()))
    plain_budget, measured_budget = session(), session()
    current_budget = [plain_budget]
    predict = CoupledLearner.predict

    def final_only(self, features):
        assert current_budget[0].updates_completed == 4
        return predict(self, features)

    monkeypatch.setattr(CoupledLearner, "predict", final_only)
    plain_renewal, measured_renewal = (
        renewal(reference, plain_budget),
        renewal(measured, measured_budget),
    )
    cap = capacity(measured, measured_budget)
    snapshots: list[Any] = []
    expected_snapshots: list[Any] = []
    expected = run(
        reference,
        plain_budget,
        plain_renewal,
        observer=lambda n, state, arrived: expected_snapshots.append(state),
    )
    current_budget[0] = measured_budget
    actual = run(
        measured,
        measured_budget,
        measured_renewal,
        cap,
        observer=lambda n, state, arrived: snapshots.append(state),
    )
    assert without_times(
        {key: value for key, value in actual.items() if key != "capacity"}
    ) == without_times(expected)
    np.testing.assert_equal([asdict(s) for s in snapshots], [asdict(s) for s in expected_snapshots])
    assert pickle.dumps((reference.snapshot_state(), measured.snapshot_state())) == before
    final = snapshots[-1]
    assert final.unit_history.unit_ids.tolist() == [3, 1, 2]
    assert final.unit_history.update_age.tolist() == [2, 4, 4]
    assert final.unit_history.updates == 4
    assert cap.observation_only is True
    cost = actual["capacity"]
    auxiliary = 72 if kind == "pc" else 120
    assert cost["completed_native_calls"] == 4 and cost["completed_training_rows"] == 8
    assert cost["mean_parameters"] == cost["peak_parameters"] == 13
    assert cost["mean_auxiliary_bytes"] == cost["peak_auxiliary_bytes"] == auxiliary
    assert cost["mean_total_array_bytes"] == cost["peak_total_array_bytes"] == 104 + auxiliary
    assert [s["phase"] for s in cost["samples"]] == [
        "initial",
        "post_wake",
        "post_wake",
        "post_consolidation",
        "post_wake",
        "post_wake",
        "post_consolidation",
    ]
    events = actual["unit_renewal"]["events"]
    assert events[0]["committed_units"] == 1 and events[1]["status"] == "allowance_exhausted"
    assert all(event["work_seconds"] >= 0 for event in events)
    assert actual["unit_renewal"]["event_seconds"] == sum(event["work_seconds"] for event in events)
    assert actual["score_forward_calls"] == 2
    assert measured_budget.updates_completed == plain_budget.updates_completed == 4


class FakeRenewable:
    def __init__(self, state=None, trace=None, flags=None, *, mature=2):
        parameters = CoupledParameters(
            (np.zeros((2, 3)),), (np.zeros((1, 3)),), np.zeros((3, 1)), np.zeros((1, 1))
        )
        self.state = (
            deepcopy(state)
            if state is not None
            else RenewableCoupledSnapshot(
                3,
                parameters,
                (1.0,),
                None,
                initial_renewal_history(3),
                RenewalPolicy(mature, 0.0, 1),
            )
        )
        self.trace = [] if trace is None else trace
        self.flags = {} if flags is None else flags

    def fork(self):
        self.trace.append("fork")
        return FakeRenewable(self.state, self.trace, self.flags)

    def snapshot_state(self):
        return deepcopy(self.state)

    def restore_state(self, state):
        self.state = deepcopy(state)

    def train_batch(self, features, targets):
        self.trace.append("train")
        current = self.state.unit_history
        assert current is not None
        history = advance_renewal_history(current, np.zeros((3, 1)))
        self.state = replace(self.state, unit_history=history)
        return TrainingDiagnostic("fake_committed_updates", float(history.updates))

    def renew_units(self, maximum):
        self.trace.append("renew")
        if self.flags.get("renew_failure"):
            raise RuntimeError("primary renewal failure")
        history = self.state.unit_history
        policy = self.state.renewal_policy
        assert history is not None and policy is not None
        indices = select_renewal_indices(history, policy, maximum)
        old = tuple(int(history.unit_ids[i]) for i in indices)
        new = tuple(range(history.next_unit_id, history.next_unit_id + len(indices)))
        ids, ages, importance = (
            history.unit_ids.copy(),
            history.update_age.copy(),
            history.importance_ema.copy(),
        )
        for index, identity in zip(indices, new):
            ids[index], ages[index], importance[index] = identity, 0, 0
        history = replace(
            history,
            unit_ids=ids,
            update_age=ages,
            importance_ema=importance,
            next_unit_id=history.next_unit_id + len(indices),
        )
        self.state = replace(self.state, unit_history=history)
        callback = self.flags.get("after_renew")
        if callback is not None:
            callback()
        return RenewalResult(indices, old, new)

    def predict(self, features):
        assert self.flags.get("score_allowed"), "heldout access before all training"
        self.trace.append("score")
        return np.full((len(features), 1), 0.5)


def direct_capacity(*, split=None, measure=measure_numpy_capacity):
    source, budget = FakeRenewable(), session()
    cap = capacity(source, budget, measure=measure, split=split)
    maximum = cap.bind(source, budget, 4)
    clock = LogicalClock()
    runtime = ActorShadowRuntime(
        source,
        actor_version="a0",
        candidate_version="c0",
        clock=clock,
        budget=budget,
        max_experiences=4,
        max_consolidations=maximum,
    )
    cap.start(runtime)
    return source, budget, cap, runtime, clock


def fake_wake(runtime, clock, cap):
    clock.advance_to(1)
    runtime.record_experience(
        Experience(
            "sample",
            "renewal",
            1,
            "a0",
            np.zeros((1, 2)),
            "train",
            ExperiencePermissions(training=True),
        )
    )
    runtime.record_label(LabelArrival("event", "sample", "renewal", 1, "a0", np.full((1, 1), 0.5)))
    assert len(runtime.train_ready(max_updates=1)) == 1
    cap.after_update(runtime, 1, 1)


@pytest.mark.parametrize("fault", ["unobserved", "duplicate", "count", "runtime", "split_mode"])
def test_should_refuse_post_consolidation_boundary_misuse_before_another_measurement(fault):
    source, budget, cap, runtime, clock = direct_capacity(
        split=2 if fault == "split_mode" else None
    )
    if fault != "unobserved":
        fake_wake(runtime, clock, cap)
    if fault == "duplicate":
        cap.after_consolidation(runtime, 1)
    before = deepcopy(cap.report)
    if fault == "runtime":
        runtime = ActorShadowRuntime(
            source,
            actor_version="a0",
            candidate_version="c0",
            clock=clock,
            budget=budget,
            max_experiences=4,
        )
    with pytest.raises(ValueError):
        cap.after_consolidation(runtime, 2 if fault == "count" else 1)
    assert cap.report == before


@pytest.mark.parametrize("fault", ["charged", "limit", "every", "clock", "budget", "observed"])
def test_should_refuse_unobserved_renewal_charge_or_binding_changes_before_the_first_train(fault):
    source, budget = FakeRenewable(), session()
    renew, cap = renewal(source, budget), capacity(source, budget)

    def mutate(updates, state, arrived):
        assert updates == 0
        if fault == "charged":
            renew._charged = 0 + 1
        elif fault == "limit":
            renew._limit = 2
        elif fault == "every":
            renew._every = 1
        elif fault == "clock":
            budget.clock = lambda: 0.0
        elif fault == "budget":
            renew._budget = session()
        else:
            renew._observed = 1

    with pytest.raises(ValueError):
        run(source, budget, renew, cap, observer=mutate)
    assert budget.updates_completed == 0
    assert "train" not in source.trace and "score" not in source.trace


@pytest.mark.parametrize("fault", ["schedule", "identity"])
def test_should_check_original_renewal_after_an_opaque_capacity_callback_before_modulo(fault):
    source, budget = FakeRenewable(), session()
    renew = renewal(source, budget)
    foreign = renewal(source, session())
    owner: list[Any] = []

    def measure(state):
        if state.unit_history.updates:
            if fault == "schedule":
                renew._every = 0
            else:
                owner[0]._renewal = foreign
        return measure_numpy_capacity(state)

    cap = capacity(source, budget, measure=measure)
    training, final = data()
    stream = AttributionStream(
        source,
        training,
        final,
        checkpoint_updates=(0, 4),
        label_delay=1,
        budget=budget,
        renewal=renew,
        capacity=cap,
    )
    owner.append(stream)
    with pytest.raises(ValueError, match="every_updates|original"):
        stream.run_until(4)
    assert budget.updates_completed == source.trace.count("train") == 1
    assert "renew" not in source.trace and "score" not in source.trace


def test_should_preserve_known_returned_work_and_secondary_error_on_the_preverified_event(
    monkeypatch,
):
    source, budget = FakeRenewable(), session(2)
    renew = renewal(source, budget, total=2)
    cap = capacity(source, budget, total=2)
    source.flags["after_renew"] = lambda: setattr(renew, "_every", 0)

    def secondary(*args, **kwargs):
        raise LookupError("secondary readback unavailable")

    monkeypatch.setattr(cap, "after_consolidation", secondary)
    with pytest.raises(ValueError, match="every_updates|original schedule"):
        run(source, budget, renew, cap, total=2)
    event = renew.report["events"][0]
    assert event["charged_units"] == event["attempted_units"] == event["returned_units"] == 1
    assert event["committed_units"] == 0 and event["receipt"] is None
    assert event["capacity_observation_error"] == "LookupError: secondary readback unavailable"
    assert source.trace.count("train") == budget.updates_completed == 2
    assert "score" not in source.trace


@pytest.mark.parametrize(
    "combination", ["split_renewal", "replay_capacity", "replay_renewal", "all_three"]
)
def test_should_refuse_unsupported_helper_pairs_before_any_binding(combination):
    source, budget = FakeRenewable(), session()
    renew = renewal(source, budget)
    cap = capacity(source, budget, split=2 if combination == "split_renewal" else None)
    replay_budget = ToyBudgetSession(ToyExecutionBudget(max_replay_examples=2), lambda: 0.0)
    replay = StreamRehearsal(
        source,
        budget,
        StreamReplayPolicy(
            2,
            1,
            ReplayRetentionBudget(4, 96),
            ReplayAllocation(ReplayRetentionBudget(2, 48), ReplayRetentionBudget(2, 48)),
            53,
        ),
        replay_budget=replay_budget,
        total_updates=4,
        input_dim=2,
    )
    selected_renew = None if combination == "replay_capacity" else renew
    selected_capacity = None if combination == "replay_renewal" else cap
    with pytest.raises(ValueError):
        run(
            source,
            budget,
            selected_renew,
            selected_capacity,
            rehearsal=None if combination == "split_renewal" else replay,
        )
    assert not renew.report["bound"] and not cap.report["bound"] and not replay.report["bound"]
    assert "fork" not in source.trace and "train" not in source.trace


@pytest.mark.parametrize("fault", ["before_publication", "after_publication", "postread_failure"])
def test_should_preserve_primary_renewal_errors_charges_and_actual_commit_receipts(
    monkeypatch, fault
):
    source, budget = FakeRenewable(), session(2)
    renew = renewal(source, budget, total=2)
    primary = RuntimeError("primary renewal failure")
    original_consolidate = ActorShadowRuntime.consolidate

    def consolidate(self, *args, **kwargs):
        original_consolidate(self, *args, **kwargs)
        raise primary

    def measure(state):
        if fault == "postread_failure" and state.unit_history.next_unit_id == 4:
            raise LookupError("secondary capacity failure")
        return measure_numpy_capacity(state)

    cap = capacity(source, budget, total=2, measure=measure)
    if fault == "before_publication":
        source.flags["renew_failure"] = True
    else:
        monkeypatch.setattr(ActorShadowRuntime, "consolidate", consolidate)
    with pytest.raises(RuntimeError, match="primary renewal failure") as error:
        run(source, budget, renew, cap, total=2)
    if fault != "before_publication":
        assert error.value is primary
    report = renew.report
    committed = int(fault != "before_publication")
    assert report["charged_units"] == report["attempted_units"] == 1
    assert report["committed_units"] == committed
    assert (report["events"][0]["receipt"] is not None) == bool(committed)
    assert budget.updates_completed == cap.report["completed_native_calls"] == 2
    assert "score" not in source.trace
    if fault == "postread_failure":
        assert (
            "LookupError: secondary capacity failure"
            == report["events"][0]["capacity_observation_error"]
        )
        assert cap.report["samples"][-1]["phase"] == "post_consolidation"
        assert cap.report["samples"][-1]["status"] == "failed"


@pytest.mark.parametrize("fault", ["raise", "changed_usage"])
def test_should_stop_failed_post_observation_without_adding_wake_weight_or_refunding_work(fault):
    armed = [False]

    def measure(state):
        if armed[0]:
            if fault == "raise":
                raise RuntimeError("post observation unavailable")
            return CapacityUsage(17, 136, 72)
        return measure_numpy_capacity(state)

    source, budget, cap, runtime, clock = direct_capacity(measure=measure)
    fake_wake(runtime, clock, cap)
    before = cap.report
    armed[0] = True
    with pytest.raises((RuntimeError, ValueError)):
        cap.after_consolidation(runtime, 1)
    report = cap.report
    assert report["stopped"] and budget.updates_completed == 1
    for name in (
        "completed_native_calls",
        "completed_training_rows",
        "mean_parameters",
        "mean_total_array_bytes",
    ):
        assert report[name] == before[name]
    assert report["samples"][-1]["phase"] == "post_consolidation"
    if fault == "raise":
        assert report["samples"][-1]["status"] == "failed"
    else:
        # Reading a changed usage can succeed while the fixed-size invariant
        # fails; preserve that measured value and the helper's failure reason.
        assert report["samples"][-1]["usage"] == asdict(CapacityUsage(17, 136, 72))
        assert report["peak_parameters"] == 17


@pytest.mark.parametrize("case", ["zero_allowance", "zero_selected"])
def test_should_observe_reached_zero_work_events_without_creating_extra_wake_cost(case):
    source, budget = FakeRenewable(mature=10 if case == "zero_selected" else 2), session()
    source.flags["score_allowed"] = True
    renew = renewal(source, budget, limit=1 if case == "zero_selected" else 0)
    cap = capacity(source, budget)
    result = run(source, budget, renew, cap)
    report = result["capacity"]
    assert report["completed_native_calls"] == source.trace.count("train") == 4
    assert report["completed_training_rows"] == 8 and report["mean_parameters"] == 13
    assert (
        len([sample for sample in report["samples"] if sample["phase"] == "post_consolidation"])
        == 2
    )
    assert result["unit_renewal"]["charged_units"] == 0
    assert result["unit_renewal"]["committed_units"] == 0
    detached = cap.report
    detached["samples"].clear()
    assert len(cap.report["samples"]) == 7


def test_should_observe_committed_renewal_after_real_clock_expiry_without_renewing_admission(
    monkeypatch,
):
    from src.app.toy_execution_budget import ToyExecutionStopped

    source = FakeRenewable()
    seconds = [0.0]
    clock = lambda: seconds[0]
    budget = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=2, max_wall_seconds=60), clock
    )
    started_at, limits = budget.started_at, budget.budget
    renew = renewal(source, budget, total=2)
    cap = capacity(source, budget, total=2)
    restore = FakeRenewable.restore_state
    admit_final = budget.before_final
    stops: list[ToyExecutionStopped] = []

    def expire_during_publication(self, state):
        restore(self, state)
        history = state.unit_history
        assert history is not None
        if history.next_unit_id == 4:
            seconds[0] = 60.0

    def record_admission_failure():
        try:
            admit_final()
        except ToyExecutionStopped as error:
            stops.append(error)
            raise

    monkeypatch.setattr(FakeRenewable, "restore_state", expire_during_publication)
    monkeypatch.setattr(budget, "before_final", record_admission_failure)
    training, final = data(2)
    stream = AttributionStream(
        source,
        training,
        final,
        checkpoint_updates=(0, 2),
        label_delay=1,
        budget=budget,
        renewal=renew,
        capacity=cap,
    )
    with pytest.raises(ToyExecutionStopped) as failure:
        stream.run_until(2)
    assert len(stops) == 1 and failure.value is stops[0]
    assert failure.value.stop.reason == "max_wall_seconds"
    assert failure.value.stop.updates_completed == 2
    event = renew.report["events"][0]
    assert event["status"] == "failed"
    assert event["charged_units"] == event["attempted_units"] == 1
    assert event["returned_units"] == event["committed_units"] == 1
    receipts = stream._runtime.applied_consolidations
    assert len(receipts) == 1 and asdict(receipts[0]) == event["receipt"]
    assert "capacity_observation_error" not in event
    post = cap.report["samples"][-1]
    assert post["phase"] == "post_consolidation" and post["status"] == "complete"
    assert post["usage"] == asdict(CapacityUsage(13, 104, 72))
    assert cap.report["completed_native_calls"] == 2
    assert cap.report["completed_training_rows"] == 4
    assert budget.budget is limits and budget.clock is clock
    assert budget.started_at == started_at == 0.0 and budget.last_clock == 60.0
    assert budget.updates_completed == source.trace.count("train") == 2
    assert budget.replay_examples_completed == 0 and "score" not in source.trace
    with pytest.raises(ValueError):
        stream.run_until(2)
    with pytest.raises(ValueError):
        stream.finish()
    assert len(stops) == 1 and source.trace.count("train") == 2
    assert budget.started_at == started_at and budget.last_clock == 60.0
