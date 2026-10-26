"""Rehearse arrived rows through actual stream/runtime boundaries and fake learners."""

from dataclasses import asdict
import json
import pickle
from typing import Any

import numpy as np
import pytest

from src.app.actor_shadow import ActorShadowRuntime
from src.app.coupled_attribution import AttributionBatch, run_attribution_stream
from src.app.stream_rehearsal import StreamRehearsal, StreamReplayPolicy
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget, ToyExecutionStopped
from src.core.circadian_predictive_coding import ReplayRetentionBudget
from src.core.learner_ports import TrainingDiagnostic
from src.core.experience import LogicalClock
from src.core.shared_replay_schedule import ReplayAllocation


def policy(**changes):
    values: dict[str, Any] = dict(
        every_updates=2,
        rows_per_event=2,
        retention=ReplayRetentionBudget(6, 144),
        allocation=ReplayAllocation(ReplayRetentionBudget(3, 72), ReplayRetentionBudget(3, 72)),
        seed=53,
    )
    values.update(changes)
    return StreamReplayPolicy(**values)


def budgets(replay=4, *, seconds=None):
    clock = (lambda: 0.0) if seconds is None else lambda: seconds[0]
    return (
        ToyBudgetSession(ToyExecutionBudget(max_training_updates=4, max_wall_seconds=10.0), clock),
        ToyBudgetSession(
            ToyExecutionBudget(max_replay_examples=replay, max_wall_seconds=10.0), clock
        ),
    )


class FakeLearner:
    """Private state forks, with an external trace of actual attempted fake work."""

    def __init__(self, control=None, state=0, *, name="source", increment=1):
        self.control: dict[str, Any] = (
            {"trace": [], "models": [], "forks": 0, "replay_calls": 0}
            if control is None
            else control
        )
        self.control["models"].append(self)
        self.state = np.array([float(state)])
        self.name, self.increment = name, increment

    def fork(self):
        self.control["forks"] += 1
        number = self.control["forks"]
        self.control["trace"].append(("fork", self.name, number))
        if number > 2 and self.control.get("fork_same"):
            return self
        return FakeLearner(
            self.control, self.state[0], name=f"fork-{number}", increment=self.increment
        )

    def snapshot_state(self):
        self.control["trace"].append(("snapshot", self.name, float(self.state[0])))
        return self.state.copy()

    def restore_state(self, state):
        self.control["trace"].append(("restore", self.name, float(state[0])))
        fault = self.control.get("restore_fault")
        if (fault == "prepared" and self.name not in ("source", "fork-1", "fork-2")) or (
            fault == "candidate" and self.name == "fork-2"
        ):
            raise self.control["error"]
        self.state = state.copy()
        if self.name == "fork-2" and self.control.get("expire_after_commit"):
            self.control["seconds"][0] = 10.0

    def train_batch(self, features, targets):
        replay = len(features) == 1
        if replay:
            self.control["replay_calls"] += 1
        charged = self.control["replay_budget"].replay_examples_completed
        self.control["trace"].append(
            (
                "train",
                self.name,
                "replay" if replay else "wake",
                charged,
                features.copy(),
                targets.copy(),
                float(self.state[0]),
            )
        )
        self.state += self.increment
        if replay and self.control.get("fail_replay") == self.control["replay_calls"]:
            raise self.control["error"]
        if self.control.get("mutate_inputs"):
            features.fill(999)
            targets.fill(0.5)
        return TrainingDiagnostic("fake_rehearsal_v1", float(self.state[0]))

    def predict(self, features):
        self.control["trace"].append(("predict", self.name, float(self.state[0]), features.copy()))
        return np.full((len(features), 1), min(0.95, (self.state[0] + 1) / 20))


def batches():
    training = tuple(
        AttributionBatch(
            "train",
            "A" if index % 2 == 0 else "B",
            (f"train-{index}-0", f"train-{index}-1"),
            np.array([[index + 0.1, 0.2], [index + 0.3, 0.4]]),
            np.full((2, 1), float(index % 2)),
        )
        for index in range(4)
    )
    heldout = tuple(
        AttributionBatch(
            "final_test",
            context,
            (f"test-{context}",),
            np.array([[90.0 + index, 91.0]]),
            np.array([[float(index)]]),
        )
        for index, context in enumerate(("A", "B"))
    )
    return training, heldout


def setup(*, replay=4, increment=1, **faults):
    seconds = [0.0]
    wake, replay_budget = budgets(replay, seconds=seconds)
    source = FakeLearner(increment=increment)
    source.control.update(
        replay_budget=replay_budget,
        seconds=seconds,
        error=RuntimeError("original rehearsal fault"),
        **faults,
    )
    rehearsal = StreamRehearsal(
        source, wake, policy(), replay_budget=replay_budget, total_updates=4, input_dim=2
    )
    return source, wake, replay_budget, rehearsal


def run(source, wake, rehearsal, *, delay=2, observe=None):
    training, heldout = batches()
    return run_attribution_stream(
        source,
        training,
        heldout,
        checkpoint_updates=(0, 2, 4),
        label_delay=delay,
        budget=wake,
        rehearsal=rehearsal,
        observe_checkpoint=observe,
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("every_updates", 0),
        ("every_updates", True),
        ("every_updates", 513),
        ("every_updates", 2.0),
        ("rows_per_event", 0),
        ("rows_per_event", True),
        ("rows_per_event", 129),
        ("seed", True),
        ("seed", -1),
        ("seed", 2**64),
    ],
)
def test_should_refuse_nonexact_and_out_of_bounds_policy_counts_before_learner_work(field, value):
    with pytest.raises(ValueError):
        policy(**{field: value})


@pytest.mark.parametrize(
    "fault",
    [
        "retention_type",
        "retention_rows",
        "retention_bytes",
        "allocation_type",
        "nested_tamper",
        "sum_rows",
        "sum_bytes",
    ],
)
def test_should_revalidate_complete_nested_retention_and_allocation_controls(fault):
    changes: dict[str, Any] = {}
    if fault == "retention_type":
        changes["retention"] = object()
    elif fault == "retention_rows":
        changes["retention"] = ReplayRetentionBudget(129, 144)
    elif fault == "retention_bytes":
        changes["retention"] = ReplayRetentionBudget(6, 1024 * 1024 + 1)
    elif fault == "allocation_type":
        changes["allocation"] = object()
    elif fault == "nested_tamper":
        value = ReplayAllocation(ReplayRetentionBudget(3, 72), ReplayRetentionBudget(3, 72))
        object.__setattr__(value.recent, "max_bytes", True)
        changes["allocation"] = value
    elif fault == "sum_rows":
        changes["retention"] = ReplayRetentionBudget(5, 144)
    else:
        changes["retention"] = ReplayRetentionBudget(6, 143)
    with pytest.raises(ValueError):
        policy(**changes)


@pytest.mark.parametrize(
    "fault",
    [
        "total_zero",
        "total_boolean",
        "total_cap",
        "dim_zero",
        "dim_boolean",
        "wake_type",
        "same_budget",
        "unlimited_replay",
        "spent_wake",
        "spent_replay",
        "wake_width",
        "replay_width",
    ],
)
def test_should_refuse_invalid_session_geometry_and_nonfresh_or_unbounded_budgets_without_fork(
    fault,
):
    source = FakeLearner()
    wake, replay = budgets()
    total: Any = 4
    dimension: Any = 2
    if fault.startswith("total"):
        total = {"total_zero": 0, "total_boolean": True, "total_cap": 513}[fault]
    elif fault.startswith("dim"):
        dimension = 0 if fault == "dim_zero" else True
    elif fault == "wake_type":
        wake = object()
    elif fault == "same_budget":
        replay = wake
    elif fault == "unlimited_replay":
        replay = ToyBudgetSession(ToyExecutionBudget(max_wall_seconds=10.0), lambda: 0.0)
    elif fault == "spent_wake":
        wake.record_update()
    elif fault == "spent_replay":
        replay.record_replay(1)
    else:
        selected = wake if fault == "wake_width" else replay
        selected.budget = ToyExecutionBudget(
            max_training_updates=4,
            max_replay_examples=None if selected is wake else 4,
            max_hidden_width=2,
        )
    with pytest.raises(ValueError):
        StreamRehearsal(
            source, wake, policy(), replay_budget=replay, total_updates=total, input_dim=dimension
        )
    assert source.control["forks"] == 0


@pytest.mark.parametrize("fault", ["learner", "budget", "total", "dimension", "reuse"])
def test_should_bind_exact_stream_identities_once_before_any_native_fork(fault):
    source, wake, _, rehearsal = setup()
    args: list[Any] = [source, wake, 4, 2]
    if fault == "reuse":
        assert rehearsal.bind(*args) == 2
    else:
        index = {"learner": 0, "budget": 1, "total": 2, "dimension": 3}[fault]
        args[index] = (
            FakeLearner() if fault == "learner" else budgets()[0] if fault == "budget" else 3
        )
    with pytest.raises(ValueError):
        rehearsal.bind(*args)
    assert source.control["forks"] == 0


@pytest.mark.parametrize("delay", [0, 2])
def test_should_rehearse_before_checkpoint_capture_keep_wake_receipts_and_defer_all_heldout_prediction(
    delay,
):
    source, wake, replay, rehearsal = setup()
    observed: list[tuple[int, float]] = []

    def observe(updates, state, batch):
        observed.append((updates, float(state[0])))
        source.control["trace"].append(("observe", updates, float(state[0])))

    result = run(source, wake, rehearsal, delay=delay, observe=observe)
    assert observed == [(0, 0.0), (2, 4.0), (4, 8.0)]
    assert result["completed_updates"] == wake.updates_completed == 4
    assert result["training_examples"] == 8 and result["score_forward_calls"] == 6
    assert wake.replay_examples_completed == 0 and replay.replay_examples_completed == 4
    assert replay.updates_completed == 0 and source.state[0] == 0
    events = source.control["trace"]
    trains = [event for event in events if event[0] == "train"]
    predictions = [event for event in events if event[0] == "predict"]
    assert [event[2] for event in trains] == [
        "wake",
        "wake",
        "replay",
        "replay",
        "wake",
        "wake",
        "replay",
        "replay",
    ]
    assert [event[3] for event in trains if event[2] == "replay"] == [1, 2, 3, 4]
    assert max(index for index, event in enumerate(events) if event[0] == "train") < min(
        index for index, event in enumerate(events) if event[0] == "predict"
    )
    assert [event[2] for event in predictions] == [0.0, 0.0, 4.0, 4.0, 8.0, 8.0]
    assert [row["updates"] for row in result["checkpoints"]] == [0, 2, 4]
    assert len(result["applied_consolidations"]) == 2
    assert [row["attempt_number"] for row in result["applied_consolidations"]] == [1, 2]
    assert [row["update_number"] for row in result["applied_updates"]] == [1, 2, 3, 4]
    assert [row["applied_at"] for row in result["applied_updates"]] == list(range(delay, delay + 4))
    training, heldout = batches()
    for event in trains:
        assert all(not np.array_equal(event[4], reserved.features) for reserved in heldout)
        if event[2] == "replay":
            assert len(event[4]) == 1
            assert any(
                np.array_equal(event[4], batch.features[index : index + 1])
                and np.array_equal(event[5], batch.targets[index : index + 1])
                for batch in training
                for index in range(2)
            )


def test_should_preserve_disabled_schema_and_call_order_when_rehearsal_is_explicitly_none():
    first, first_budget, _, _ = setup()
    second, second_budget, _, _ = setup()
    training, heldout = batches()
    implicit = run_attribution_stream(
        first, training, heldout, budget=first_budget, checkpoint_updates=(0, 2, 4), label_delay=2
    )
    explicit = run_attribution_stream(
        second,
        training,
        heldout,
        budget=second_budget,
        rehearsal=None,
        checkpoint_updates=(0, 2, 4),
        label_delay=2,
    )
    assert implicit == explicit
    assert set(implicit) == {
        "completed_updates",
        "training_examples",
        "score_forward_calls",
        "checkpoints",
        "applied_updates",
    }
    np.testing.assert_equal(first.control["trace"], second.control["trace"])
    assert sum(event[0] == "train" for event in first.control["trace"]) == 4


def test_should_propose_identical_rehearsal_rows_across_fake_methods_with_different_native_state_dynamics():
    plans = []
    for increment in (1, 2, 3, 4):
        source, wake, _, rehearsal = setup(increment=increment)
        run(source, wake, rehearsal)
        plans.append(
            tuple(
                (event[4].tobytes(), event[5].tobytes())
                for event in source.control["trace"]
                if event[0] == "train" and event[2] == "replay"
            )
        )
        assert source.state[0] == 0
    assert len(plans[0]) == 4 and all(plan == plans[0] for plan in plans)


def test_should_copy_arrived_training_and_heldout_arrays_before_adversarial_fake_training_and_observers():
    source, wake, _, rehearsal = setup(mutate_inputs=True)
    training, heldout = batches()
    before = pickle.dumps((training, heldout))

    def mutate(updates, state, batch):
        state.fill(888)
        if batch is not None:
            batch.features.fill(777)
            batch.targets.fill(0.5)

    result = run_attribution_stream(
        source,
        training,
        heldout,
        checkpoint_updates=(0, 2, 4),
        label_delay=2,
        budget=wake,
        rehearsal=rehearsal,
        observe_checkpoint=mutate,
    )
    assert pickle.dumps((training, heldout)) == before
    assert source.state[0] == 0 and result["completed_updates"] == 4
    trains = [event for event in source.control["trace"] if event[0] == "train"]
    assert all(np.max(event[4]) < 10 for event in trains)
    assert [event[2] for event in source.control["trace"] if event[0] == "predict"] == [
        0.0,
        0.0,
        4.0,
        4.0,
        8.0,
        8.0,
    ]


@pytest.mark.parametrize("value", [False, True, object()])
def test_should_refuse_nonhelper_rehearsal_before_runtime_forks(value):
    source, wake, _, _ = setup()
    with pytest.raises(ValueError):
        run(source, wake, value)
    assert source.control["forks"] == 0


def test_should_own_detached_complete_policy_and_bind_the_original_planned_event_count():
    source = FakeLearner()
    wake, replay = budgets()
    controls = policy()
    expected = asdict(controls)
    source.control["replay_budget"] = replay
    rehearsal = StreamRehearsal(
        source, wake, controls, replay_budget=replay, total_updates=4, input_dim=2
    )
    object.__setattr__(controls, "every_updates", 1)
    object.__setattr__(controls, "seed", 17)
    object.__setattr__(controls.allocation.recent, "max_bytes", 1)
    assert rehearsal.report["policy"] == expected
    result = run(source, wake, rehearsal)
    assert rehearsal.report["planned_events"] == 2
    assert len(result["applied_consolidations"]) == 2
    assert rehearsal.report["policy"] == expected


@pytest.mark.parametrize("cap,completed_events,wakes", [(1, 0, 2), (3, 1, 4)])
def test_should_refuse_a_whole_event_when_its_remaining_replay_budget_cannot_pay_all_rows(
    cap, completed_events, wakes
):
    source, wake, replay, rehearsal = setup(replay=cap)
    with pytest.raises(ToyExecutionStopped):
        run(source, wake, rehearsal)
    report = rehearsal.report
    assert report["stopped"] is True
    assert wake.updates_completed == wakes
    assert report["charged_exposures"] == replay.replay_examples_completed == 2 * completed_events
    assert (
        report["attempted_updates"]
        == report["returned_updates"]
        == report["committed_updates"]
        == 2 * completed_events
    )
    assert [event["status"] for event in report["events"]] == (
        ["failed", "not_run"] if cap == 1 else ["complete", "failed"]
    )
    failed = report["events"][completed_events]
    assert failed["charged_exposures"] == failed["attempted_updates"] == 0
    assert failed["returned_updates"] == failed["committed_updates"] == 0
    assert source.control["replay_calls"] == 2 * completed_events
    assert not any(event[0] == "predict" for event in source.control["trace"])
    assert source.state[0] == 0


@pytest.mark.parametrize("failed_call", [1, 2])
def test_should_charge_before_a_replay_call_that_mutates_then_raises_without_committing_the_transform(
    failed_call,
):
    source, wake, replay, rehearsal = setup(fail_replay=failed_call)
    error = source.control["error"]
    with pytest.raises(RuntimeError) as caught:
        run(source, wake, rehearsal)
    assert caught.value is error
    report = rehearsal.report
    assert report["stopped"] is True and report["bound"] is True
    assert (
        report["attempted_updates"]
        == report["charged_exposures"]
        == replay.replay_examples_completed
        == failed_call
    )
    assert report["returned_updates"] == failed_call - 1
    assert report["committed_updates"] == 0 and wake.updates_completed == 2
    assert wake.replay_examples_completed == 0 and replay.updates_completed == 0
    first, second = report["events"]
    assert first["status"] == "failed" and second["status"] == "not_run"
    assert first["receipt"] is None and first["committed_updates"] == 0
    assert len(first["diagnostics"]) == failed_call - 1
    calls = [
        event for event in source.control["trace"] if event[0] == "train" and event[2] == "replay"
    ]
    assert [event[3] for event in calls] == list(range(1, failed_call + 1))
    assert (
        source.control["models"][2].state[0] == 2
    )  # Original candidate, not failed private transform.
    assert source.state[0] == 0
    assert not any(event[0] == "predict" for event in source.control["trace"])


@pytest.mark.parametrize("fault", ["fork_same", "prepared", "candidate"])
def test_should_preserve_charges_and_no_scores_when_fork_restore_or_publication_fails(fault):
    options = {"fork_same": True} if fault == "fork_same" else {"restore_fault": fault}
    source, wake, replay, rehearsal = setup(**options)
    with pytest.raises((RuntimeError, ValueError)) as caught:
        run(source, wake, rehearsal)
    if fault != "fork_same":
        assert caught.value is source.control["error"]
    report = rehearsal.report
    expected = 2 if fault == "candidate" else 0
    assert (
        report["attempted_updates"]
        == report["returned_updates"]
        == report["charged_exposures"]
        == expected
    )
    assert replay.replay_examples_completed == expected and report["committed_updates"] == 0
    assert report["events"][0]["status"] == "failed"
    assert report["events"][1]["status"] == "not_run"
    assert report["events"][0]["receipt"] is None
    assert wake.updates_completed == 2 and source.state[0] == 0
    assert not any(event[0] == "predict" for event in source.control["trace"])


def test_should_report_actual_committed_receipt_when_the_post_commit_budget_check_raises():
    source, wake, replay, rehearsal = setup(expire_after_commit=True)
    with pytest.raises(ToyExecutionStopped) as caught:
        run(source, wake, rehearsal)
    assert caught.value.stop.reason == "max_wall_seconds"
    report = rehearsal.report
    assert report["stopped"] is True and report["events"][0]["status"] == "failed"
    assert (
        report["charged_exposures"]
        == report["attempted_updates"]
        == report["returned_updates"]
        == report["committed_updates"]
        == 2
    )
    assert replay.replay_examples_completed == 2
    assert report["events"][0]["receipt"] is not None
    assert report["events"][0]["committed_updates"] == 2
    assert report["events"][1]["status"] == "not_run"
    assert source.control["models"][2].state[0] == 4
    assert source.state[0] == 0 and not any(
        event[0] == "predict" for event in source.control["trace"]
    )


def test_should_keep_planned_report_rows_detached_bounded_and_data_only():
    source, wake, replay, rehearsal = setup()
    report = rehearsal.report
    assert report["bound"] is False and report["stopped"] is False
    assert report["total_updates"] == 4 and report["input_dim"] == 2
    assert report["planned_events"] == len(report["events"]) == 2
    assert [row["updates"] for row in report["events"]] == [2, 4]
    assert all(row["status"] == "not_run" and row["selected_ids"] == [] for row in report["events"])
    report["events"].clear()
    report["policy"]["allocation"]["recent"]["max_examples"] = 999
    assert len(rehearsal.report["events"]) == 2
    result = run(source, wake, rehearsal)
    expected = rehearsal.report
    assert result["rehearsal"] == expected
    assert expected["wake_updates_observed"] == 4
    assert expected["replay_budget_charged_exposures"] == replay.replay_examples_completed == 4
    assert (
        expected["charged_exposures"]
        == expected["attempted_updates"]
        == expected["returned_updates"]
        == expected["committed_updates"]
        == 4
    )
    assert all(
        row["status"] == "complete" and len(row["selected_ids"]) == 2 for row in expected["events"]
    )
    assert [row["receipt"] for row in expected["events"]] == result["applied_consolidations"]
    returned = rehearsal.report
    returned["events"][0]["selected_ids"].append("forged")
    returned["events"][0]["receipt"]["event_id"] = "forged"
    returned["reason"] = "forged"
    assert rehearsal.report == expected
    assert all(
        type(sample_id) is str and len(sample_id) == 64
        for row in expected["events"]
        for sample_id in row["selected_ids"]
    )
    json.dumps(expected, allow_nan=False)


def test_should_allow_a_predeclared_cadence_with_no_due_events_without_extra_forks_or_charges():
    source = FakeLearner()
    wake, replay = budgets()
    source.control["replay_budget"] = replay
    rehearsal = StreamRehearsal(
        source, wake, policy(every_updates=5), replay_budget=replay, total_updates=4, input_dim=2
    )
    result = run(source, wake, rehearsal)
    assert rehearsal.report["planned_events"] == 0 and rehearsal.report["events"] == []
    assert result["applied_consolidations"] == [] and replay.replay_examples_completed == 0
    assert sum(event[0] == "train" for event in source.control["trace"]) == 4
    assert source.control["forks"] == 3  # Actor/candidate, then the ordinary final scorer.


def test_should_refuse_a_fabricated_wake_count_without_observing_data_or_spending_replay():
    source, wake, replay, rehearsal = setup()
    assert rehearsal.bind(source, wake, 4, 2) == 2
    runtime = ActorShadowRuntime(
        source,
        actor_version="a0",
        candidate_version="c0",
        clock=LogicalClock(),
        budget=wake,
        max_experiences=4,
        max_consolidations=2,
    )
    before = rehearsal.report
    training, _ = batches()
    with pytest.raises(ValueError, match="actual completed wake"):
        rehearsal.after_update(runtime, 1, training[0].features, training[0].targets)
    assert rehearsal.report == before
    assert wake.updates_completed == replay.replay_examples_completed == 0
    assert source.control["forks"] == 2
    assert not any(event[0] == "train" for event in source.control["trace"])
