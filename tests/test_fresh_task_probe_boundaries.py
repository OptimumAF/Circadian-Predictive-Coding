"""Source-boundary guards preserve actual probe work and legacy helper defaults."""

from copy import deepcopy
import pickle
from typing import Any

import numpy as np
import pytest

import src.adapters.key_door_learnability as helper
import src.app.fresh_task_probe as probe
from test_checkpoint_probe_decision_route import (
    run as run_checkpoint,
    runner_for,
    setup as setup,
    slot,
)
from test_fresh_task_probe import FakeLearner, batches, run, session


@pytest.mark.parametrize("value", [False, 0, 1, "guard", np.bool_(True)])
def test_should_refuse_noncallable_guard_before_probe_inputs_clocks_or_source(value):
    source = FakeLearner()
    with pytest.raises(ValueError, match="callable or None"):
        run(source, training=(), check_boundary=value)
    assert source.trace == []


@pytest.mark.parametrize("event", ["fork", "restore", "predict", "train"])
def test_should_stop_at_corrupted_source_boundary_and_preserve_completed_work(event):
    source = FakeLearner()

    def corrupt(*args):
        source.value[0] = -1
        if event == "predict":
            return np.full((len(args[1]), 1), 0.5)
        return None

    source.hooks[event] = corrupt
    budget = session()

    def guard():
        if source.value[0] != 99:
            raise ValueError("benchmark source changed")

    result = run(source, budget=budget, check_boundary=guard)
    assert result["status"] == "failed" and "benchmark source changed" in result["reason"]
    names = [entry[0] for entry in source.trace]
    assert (
        names
        == {
            "fork": ["fork"],
            "restore": ["fork", "restore"],
            "predict": ["fork", "restore", "predict"],
            "train": ["fork", "restore", "predict", "train"],
        }[event]
    )
    committed = 1 if event == "train" else 0
    assert result["completed_updates"] == result["attempted_updates"] == committed
    assert result["completed_training_examples"] == committed
    assert budget.updates_completed == committed
    returned = 1 if event in ("train", "predict") else 0
    assert result["score_forward_calls_attempted"] == returned
    assert result["score_forward_calls_completed"] == returned
    assert [point["updates"] for point in result["checkpoints"]] == ([0] if committed else [])


def test_should_refuse_pre_fork_guard_without_counting_unattempted_native_work():
    source = FakeLearner()

    def refuse():
        raise ValueError("preflight boundary refused")

    result = run(source, check_boundary=refuse)
    assert result["status"] == "failed" and "boundary refused" in result["reason"]
    assert source.trace == []
    assert result["completed_updates"] == result["score_forward_calls_attempted"] == 0


def test_should_refuse_corruption_during_post_score_clock_before_publishing_checkpoint():
    source = FakeLearner()

    def timer():
        if sum(entry[0] == "predict" for entry in source.trace) == 3:
            source.value[0] = -1
        return 2.5

    def guard():
        if source.value[0] != 99:
            raise ValueError("post-score source changed")

    budget = session()
    result = run(source, timer=timer, budget=budget, check_boundary=guard)
    assert result["status"] == "failed" and "post-score source changed" in result["reason"]
    assert result["completed_updates"] == budget.updates_completed == 2
    assert result["score_forward_calls_completed"] == 3
    assert [point["updates"] for point in result["checkpoints"]] == [0, 1]


def test_should_guard_completion_readout_after_training_delegate_returns(monkeypatch):
    source = FakeLearner()
    original = probe._train

    def corrupt_after_training(*args, **kwargs):
        original(*args, **kwargs)
        source.value[0] = -1

    def guard():
        if source.value[0] != 99:
            raise ValueError("completion source changed")

    monkeypatch.setattr(probe, "_train", corrupt_after_training)
    budget = session()
    result = run(source, budget=budget, check_boundary=guard)
    assert result["status"] == "failed" and "completion source changed" in result["reason"]
    assert result["completed_updates"] == budget.updates_completed == 2
    assert [point["updates"] for point in result["checkpoints"]] == [0, 1, 2]
    assert result["score_forward_calls_completed"] == 3


def test_should_preserve_default_delegate_kwargs_reports_native_calls_and_timers(monkeypatch):
    score, train = probe._score, probe._train
    forwarded = []

    def score_spy(*args, **kwargs):
        forwarded.append(("score", kwargs.copy()))
        return score(*args, **kwargs)

    def train_spy(*args, **kwargs):
        forwarded.append(("train", kwargs.copy()))
        return train(*args, **kwargs)

    monkeypatch.setattr(probe, "_score", score_spy)
    monkeypatch.setattr(probe, "_train", train_spy)
    reports, traces, options, timer_counts = [], [], [], []
    for selected in ("omitted", None, "guard"):
        source, checks = FakeLearner(), []
        timers: list[None] = []

        def guard():
            checks.append(len(source.trace))

        def timer():
            timers.append(None)
            return 2.5

        kwargs = (
            {}
            if selected == "omitted"
            else {"check_boundary": guard if selected == "guard" else None}
        )
        forwarded.clear()
        reports.append(run(source, timer=timer, **kwargs))
        traces.append(pickle.dumps(source.trace))
        options.append(deepcopy(forwarded))
        timer_counts.append(len(timers))
        if selected == "guard":
            assert len(checks) > 10
            assert all(entry[1]["check_boundary"] is guard for entry in forwarded)
        else:
            assert checks == []
            assert all(set(entry[1]) == {"decision_diagnostics"} for entry in forwarded)
    assert reports[0] == reports[1] == reports[2]
    assert traces[0] == traces[1] == traces[2]
    assert timer_counts[0] == timer_counts[1] == timer_counts[2]
    assert options[0] == options[1]


class TupleSubclass(tuple):
    pass


@pytest.mark.parametrize(
    "points",
    [
        [],
        (),
        (0,),
        (0, 8, 192),
        (1, 192),
        (0, 0),
        (0, 513),
        (False, 192),
        (0, True),
        (0, 192.0),
        (0, np.int64(192)),
        TupleSubclass((0, 192)),
    ],
)
def test_should_refuse_malformed_slot_points_without_resources(points):
    with pytest.raises(ValueError, match="exact tuple"):
        helper.probe_slots(checkpoint_updates=points)


def test_should_support_explicit_return_endpoint_and_preserve_default_owned_slots():
    assert helper.probe_slots() == helper.probe_slots(checkpoint_updates=(0, 8))
    slots = helper.probe_slots(checkpoint_updates=(0, 192))
    assert [entry["checkpoint_updates"] for entry in slots] == [0, 192]
    assert all(entry["status"] == "not_run" and entry["charged_updates"] == 0 for entry in slots)
    slots[0]["reason"] = "changed"
    assert slots[1]["reason"] == "not reached"
    assert helper.probe_slots(checkpoint_updates=(0, 192))[0]["reason"] == "not reached"


@pytest.mark.parametrize("selected", ["omitted", None, "guard"])
def test_should_forward_only_supplied_checkpoint_guard_without_outer_clock_observations(
    setup, selected
):
    wake_binding, replay_binding = helper._bindings(setup.wake, setup.replay)
    checks: list[None] = []

    def guard():
        checks.append(None)

    kwargs = (
        {} if selected == "omitted" else {"check_boundary": guard if selected == "guard" else None}
    )
    run_checkpoint(setup, runner_for(setup), **kwargs)
    expected = {"budget", "checkpoint_updates"} | (
        {"check_boundary"} if selected == "guard" else set()
    )
    assert set(setup.calls[0]) == expected
    if selected == "guard":
        assert setup.calls[0]["check_boundary"] is guard and checks
    else:
        assert checks == []
    assert slot(setup)["status"] == "complete" and slot(setup)["charged_updates"] == 16
    assert helper._bindings(setup.wake, setup.replay) == (wake_binding, replay_binding)


def test_should_refuse_invalid_checkpoint_guard_before_slot_clock_snapshot_or_budget(setup):
    setup.row = {}
    before = setup.clock.calls
    invalid: Any = 1
    with pytest.raises(ValueError, match="callable or None"):
        run_checkpoint(setup, runner_for(setup), check_boundary=invalid)
    assert setup.clock.calls == before and setup.source.snapshots == 0
    assert setup.calls == setup.budgets == setup.attachments == []


def test_should_retain_full_runner_report_and_charges_on_post_runner_guard_refusal(setup):
    returned = []
    original_runner = runner_for(setup)

    def runner(*args, **kwargs):
        report = original_runner(*args, **kwargs)
        returned.append(True)
        return report

    def guard():
        if returned:
            raise ValueError("post-run boundary changed")

    with pytest.raises(ValueError, match="post-run boundary changed"):
        run_checkpoint(setup, runner, check_boundary=guard)
    result = slot(setup)
    assert result["status"] == "failed" and result["charged_updates"] == 16
    assert result["report"]["completed_updates"] == 16
    assert result["outcomes"] is None
    assert "post-run boundary changed" in result["reason"]


def test_should_preserve_source_data_under_successful_boundary_observation():
    source, state = FakeLearner(), np.array([0.0])
    training, evaluation = batches()
    before = pickle.dumps((source.value, source.policy, state, training, evaluation))

    def guard():
        assert pickle.dumps((source.value, source.policy, state, training, evaluation)) == before

    result = run(
        source, state=state, training=training, evaluation=evaluation, check_boundary=guard
    )
    assert result["status"] == "complete"
    assert pickle.dumps((source.value, source.policy, state, training, evaluation)) == before
