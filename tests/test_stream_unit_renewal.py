"""Owned native renewal through ordinary stream/runtime boundaries."""

from dataclasses import replace
import pickle
from typing import Any

import numpy as np
import pytest

from src.adapters.coupled_learners import CoupledLearner
from src.app.actor_shadow import ActorShadowRuntime
from src.app.coupled_attribution import AttributionBatch, run_attribution_stream
from src.app.stream_unit_renewal import StreamUnitRenewal
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget, ToyExecutionStopped
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork
from src.core.coupled_unit_renewal import RenewalPolicy
from src.core.experience import LogicalClock
from src.core.renewable_coupled_predictive_coding import (
    RenewableCoupledPredictiveCodingNetwork,
)


def setup(*, total=4, limit=1, every=2):
    model = RenewableCoupledPredictiveCodingNetwork(
        2, (3,), seed=17, renewal_policy=RenewalPolicy(2, 0.0, 1)
    )
    learner = CoupledLearner(
        model, learning_rate=0.02, inference_steps=2, inference_learning_rate=0.04
    )
    seconds = [0.0]
    budget = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=total, max_wall_seconds=10.0), lambda: seconds[0]
    )
    helper = StreamUnitRenewal(
        learner, budget, every_updates=every, total_updates=total, max_replacements=limit
    )
    return model, learner, budget, helper, seconds


def run(learner, budget, helper, *, total=4, observer=None, **kwargs):
    training = tuple(
        AttributionBatch(
            "train", "A", (f"train-{k}-0", f"train-{k}-1"), np.zeros((2, 2)), np.full((2, 1), 0.5)
        )
        for k in range(total)
    )
    heldout = (AttributionBatch("final_test", "A", ("final-0",), np.ones((1, 2)), np.ones((1, 1))),)
    return run_attribution_stream(
        learner,
        training,
        heldout,
        checkpoint_updates=(0, total),
        label_delay=1,
        budget=budget,
        renewal=helper,
        observe_checkpoint=observer,
        **kwargs,
    )


def test_should_renew_only_candidate_and_keep_permanent_allowance_on_later_events():
    model, learner, budget, helper, _ = setup()
    original = pickle.dumps(learner.snapshot_state())
    snapshots: list[Any] = []
    report = run(learner, budget, helper, observer=lambda n, s, b: snapshots.append(s))
    final = snapshots[-1]
    assert final.unit_history.unit_ids.tolist() == [3, 1, 2]
    assert final.unit_history.update_age.tolist() == [2, 4, 4]
    assert final.unit_history.updates == 4 and final.unit_history.next_unit_id == 4
    assert pickle.dumps(learner.snapshot_state()) == original
    assert model.read_unit_history().updates == 0
    assert report["unit_renewal"]["charged_units"] == 1
    assert report["unit_renewal"]["committed_units"] == 1
    assert len(report["applied_consolidations"]) == 1
    assert helper.report["events"][1]["status"] == "allowance_exhausted"
    assert budget.updates_completed == 4 and budget.replay_examples_completed == 0
    detached = helper.report
    detached["events"][0]["new_unit_ids"].clear()
    assert helper.report["events"][0]["new_unit_ids"] == [3]


@pytest.mark.parametrize(
    "field,value",
    [
        ("every_updates", 0),
        ("every_updates", True),
        ("total_updates", 513),
        ("total_updates", 1.0),
        ("max_replacements", -1),
        ("max_replacements", 129),
        ("max_replacements", True),
    ],
)
def test_should_refuse_invalid_schedule_before_native_work(field, value):
    _, learner, budget, _, _ = setup()
    values = dict(every_updates=2, total_updates=4, max_replacements=1)
    values[field] = value
    with pytest.raises(ValueError):
        StreamUnitRenewal(learner, budget, **values)


def test_should_refuse_ordinary_model_and_native_reset_on_ordinary_adapter():
    learner = CoupledLearner(
        CoupledPredictiveCodingNetwork(2, (3,), seed=17),
        learning_rate=0.02,
        inference_steps=2,
        inference_learning_rate=0.04,
    )
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=2), lambda: 0.0)
    with pytest.raises(ValueError, match="renewable"):
        StreamUnitRenewal(learner, budget, every_updates=2, total_updates=2, max_replacements=1)
    with pytest.raises(ValueError, match="renewable"):
        learner.renew_units(1)


def test_should_bind_once_to_original_learner_budget_and_count():
    _, learner, budget, helper, _ = setup()
    with pytest.raises(ValueError, match="original"):
        helper.bind(learner, budget, 3)
    assert helper.bind(learner, budget, 4) == 2
    with pytest.raises(ValueError, match="single-use"):
        helper.bind(learner, budget, 4)


def test_should_skip_all_resets_when_born_with_zero_allowance(monkeypatch):
    _, learner, budget, helper, _ = setup(limit=0, total=2)
    monkeypatch.setattr(CoupledLearner, "renew_units", lambda *args: pytest.fail("reset forbidden"))
    result = run(learner, budget, helper, total=2)
    assert result["unit_renewal"]["charged_units"] == 0
    assert result["applied_consolidations"] == []


@pytest.mark.parametrize("failure", ["fork", "reset", "restore"])
def test_should_keep_precharge_and_stop_after_failed_preparation_or_native_reset(
    monkeypatch, failure
):
    _, learner, budget, helper, _ = setup(total=2)
    source = pickle.dumps(learner.snapshot_state())
    original_fork = CoupledLearner.fork
    forks = [0]

    def fork(self):
        forks[0] += 1
        if forks[0] == 3 and failure == "fork":
            raise RuntimeError("precharged fault")
        copied = original_fork(self)
        if forks[0] == 3:
            if failure == "reset":
                monkeypatch.setattr(
                    copied,
                    "renew_units",
                    lambda n: (_ for _ in ()).throw(RuntimeError("precharged fault")),
                )
            if failure == "restore":
                monkeypatch.setattr(
                    copied,
                    "restore_state",
                    lambda s: (_ for _ in ()).throw(RuntimeError("precharged fault")),
                )
        return copied

    monkeypatch.setattr(CoupledLearner, "fork", fork)
    with pytest.raises(RuntimeError, match="precharged fault"):
        run(learner, budget, helper, total=2)
    result = helper.report
    assert result["charged_units"] == 1 and result["committed_units"] == 0
    assert result["stopped"] and result["events"][0]["status"] == "failed"
    learner.restore_state(pickle.loads(source))
    assert helper.report["charged_units"] == 1
    with pytest.raises(ValueError, match="single-use"):
        helper.bind(learner, budget, 2)


def test_should_preserve_committed_receipt_when_final_budget_check_fails(monkeypatch):
    _, learner, budget, helper, seconds = setup(total=2)
    original = CoupledLearner.restore_state

    def restore(self, state):
        original(self, state)
        if state.unit_history.next_unit_id == 4:
            seconds[0] = 10.0

    monkeypatch.setattr(CoupledLearner, "restore_state", restore)
    with pytest.raises(ToyExecutionStopped):
        run(learner, budget, helper, total=2)
    # The private preparation restore still has next_id3. Only publication of
    # the renewed state expires the clock, after the runtime can issue a receipt.
    event = helper.report["events"][0]
    assert event["status"] == "failed" and event["receipt"] is not None
    assert event["charged_units"] == event["returned_units"] == event["committed_units"] == 1


def test_should_refuse_unobserved_wakes_before_renewal():
    _, learner, budget, helper, _ = setup(total=2)
    helper.bind(learner, budget, 2)
    runtime = ActorShadowRuntime(
        learner,
        actor_version="a0",
        candidate_version="c0",
        clock=LogicalClock(),
        budget=budget,
        max_experiences=2,
        max_consolidations=1,
    )
    with pytest.raises(ValueError, match="actual completed"):
        helper.after_update(runtime, 1)
    assert helper.report["charged_units"] == 0


def test_should_keep_old_report_shape_when_option_is_absent():
    _, learner, budget, _, _ = setup(total=2)
    result = run(learner, budget, None, total=2)
    assert set(result) == {
        "completed_updates",
        "training_examples",
        "score_forward_calls",
        "checkpoints",
        "applied_updates",
    }


def test_should_refuse_rehearsal_composition_before_binding():
    _, learner, budget, helper, _ = setup(total=2)
    with pytest.raises(ValueError, match="simultaneous"):
        run(learner, budget, helper, total=2, rehearsal=object())
    assert not helper.report["bound"]


def test_should_refuse_replaced_selection_even_when_native_reset_returns(monkeypatch):
    _, learner, budget, helper, _ = setup(total=2)
    original = CoupledLearner.renew_units

    def renew(self, limit):
        result = original(self, limit)
        return replace(result, old_unit_ids=(999,))

    monkeypatch.setattr(CoupledLearner, "renew_units", renew)
    with pytest.raises(ValueError, match="precharged selection"):
        run(learner, budget, helper, total=2)
    assert helper.report["charged_units"] == helper.report["returned_units"] == 1
    assert helper.report["committed_units"] == 0
