"""Fitted choice behavior and causal full-source counterfactual admission."""

from copy import deepcopy
from dataclasses import replace
import json
from typing import Any, cast

import pytest

from src.app.counterfactual_controller import CounterfactualController, CounterfactualReturn
from src.core.actor_ports import VersionedState
from src.core.contextual_controller import (
    ContextualChoice,
    ContextualEstimate,
    FittedController,
    choose_contextual,
    fit_choices,
)
from src.core.controller_objective import ControllerMeasures, ControllerObjective
from src.core.experience import LogicalClock
from src.core.reference_controller import ArrivedTrainingFeedback, ReferenceObservation


def observation(clock, context="mistakes", version="original"):
    return ReferenceObservation(
        clock.now(),
        1,
        1,
        1,
        version,
        ArrivedTrainingFeedback(
            "train", "sample", version, 0, clock.now(), int(context == "mistakes"), 6
        ),
    )


def measures(gain):
    return ControllerMeasures(0, gain, 1, 1, 1, 1, 1.0, 1024, 0, True)


def setup():
    source = [VersionedState("original", {"weights": [1], "optimizer": [2], "clock": 96})]
    clock = LogicalClock()
    trainer = CounterfactualController(
        lambda: deepcopy(source[0]), json.dumps, ControllerObjective(), clock
    )
    return source, clock, trainer


def operation(clock, gain, *, source_change=None, result_change=None):
    def run(request, snapshot):
        snapshot["optimizer"][0] = 99
        clock.advance_to(request.started_at + 2)
        if source_change:
            source_change()
        result = CounterfactualReturn(request, request.started_at + 1, clock.now(), measures(gain))
        return result_change(result) if result_change else result

    return run


def trained():
    source, clock, trainer = setup()
    for context, gain in (("mistakes", 0), ("mistakes", 1), ("clean", 0), ("clean", 0)):
        trainer.run(observation(clock, context), operation(clock, gain))
    return source, clock, trainer


def test_fit_learns_opposite_preferences_and_uses_fixed_wait_tie_break():
    source, clock, trainer = trained()
    fitted = trainer.freeze()
    assert fitted.choices == ContextualChoice("wait", "wake1")
    assert fitted.fitted_at == 8 and len(fitted.estimates) == 4
    assert choose_contextual(fitted, observation(clock)).action_id == "wake1"
    assert choose_contextual(fitted, observation(clock, "clean")).action_id == "wait"
    assert source[0].state["optimizer"] == [2]
    assert trainer.snapshot()["attempts"] == trainer.snapshot()["completed"] == 4
    assert trainer.snapshot()["discarded_trials"] == 4
    with pytest.raises(ValueError, match="already fitted"):
        trainer.freeze()
    with pytest.raises(ValueError, match="closed"):
        trainer.run(observation(clock), lambda *a: pytest.fail("fifth trial"))


@pytest.mark.parametrize("fault", ["missing", "duplicate", "nonfinite", "samples"])
def test_incomplete_or_invalid_arm_evidence_cannot_fit(fault):
    rows = tuple(
        ContextualEstimate(c, a, -1.0) for c in ("clean", "mistakes") for a in ("wait", "wake1")
    )
    if fault == "missing":
        rows = rows[:3]
    if fault == "duplicate":
        rows = rows[:3] + (rows[0],)
    with pytest.raises(ValueError):
        if fault == "nonfinite":
            rows = rows[:3] + (replace(rows[3], mean_utility=float("nan")),)
        if fault == "samples":
            rows = rows[:3] + (replace(rows[3], samples=2),)
        fit_choices(rows)


@pytest.mark.parametrize("fault", ["missing_feedback", "no_ready", "no_credit"])
def test_fitted_policy_cannot_spend_without_arrived_context_and_original_credit(fault):
    _, clock, trainer = trained()
    policy = trainer.freeze()
    o = observation(clock)
    if fault == "missing_feedback":
        o = replace(o, feedback=None)
    elif fault == "no_ready":
        o = replace(o, ready_updates=0)
    else:
        o = replace(o, remaining_wake=0)
    assert choose_contextual(policy, o).action_id == "wait"


def test_fitted_policy_rejects_stale_observation_and_mutated_empirical_choices():
    _, clock, trainer = trained()
    fitted = trainer.freeze()
    with pytest.raises(ValueError, match="stale"):
        choose_contextual(fitted, observation(clock, version="replacement"))
    object.__setattr__(fitted, "choices", ContextualChoice("wake1", "wait"))
    with pytest.raises(ValueError, match="choices differ"):
        choose_contextual(fitted, observation(clock))
    assert trainer.snapshot()["fitted"].choices == ContextualChoice("wait", "wake1")


@pytest.mark.parametrize("role", ["probe_train", "inner_guard", "outer_selection", "final_test"])
def test_non_train_feedback_never_enters_fitted_selection(role):
    with pytest.raises(ValueError, match="TRAIN"):
        ArrivedTrainingFeedback(role, "s", "original", 0, 0, 0, 1)


@pytest.mark.parametrize("change", ["version", "weights", "optimizer", "clock"])
def test_source_change_before_trial_is_rejected_without_callback_or_native_work(change):
    source, clock, trainer = setup()
    if change == "version":
        source[0] = replace(source[0], actor_version="new")
    elif change == "clock":
        source[0].state[change] += 1
    else:
        source[0].state[change][0] += 1
    with pytest.raises(ValueError, match="stale"):
        trainer.run(observation(clock), lambda *a: pytest.fail("read replaced source"))
    assert trainer.snapshot()["attempts"] == 0


@pytest.mark.parametrize(
    "fault", ["source", "copy", "role", "future_label", "future_return", "incomplete"]
)
def test_stale_or_invalid_return_preserves_attempt_and_cannot_read_poison_metrics(fault):
    source, clock, trainer = setup()

    class Poison:
        def __getattribute__(self, key):
            pytest.fail("metrics accessed before stale/metadata rejection")

    def run(request, state):
        clock.advance_to(2)
        if fault == "source":
            source[0].state["optimizer"][0] += 1
        if fault == "role":
            object.__setattr__(request, "role", "final_test")
        return CounterfactualReturn(
            deepcopy(request) if fault == "copy" else request,
            3 if fault == "future_label" else 1,
            3 if fault == "future_return" else 2,
            replace(measures(0), work_complete=False)
            if fault == "incomplete"
            else cast(Any, Poison()),
        )

    with pytest.raises(ValueError):
        trainer.run(observation(clock), run)
    state = trainer.snapshot()
    assert state["attempts"] == state["discarded_trials"] == 1
    assert state["completed"] == 0 and state["stopped"]
    with pytest.raises(ValueError):
        trainer.run(observation(clock), lambda *a: pytest.fail("retry paid failed trial"))


def test_reentrant_fit_or_trial_cannot_open_another_source_or_fit_partial_table():
    _, clock, trainer = setup()
    with pytest.raises(ValueError, match="four complete"):
        trainer.freeze()

    def run(request, state):
        with pytest.raises(ValueError, match="busy"):
            trainer.freeze()
        with pytest.raises(ValueError, match="busy"):
            trainer.run(observation(clock), lambda *a: pytest.fail("nested trial"))
        clock.advance_to(2)
        return CounterfactualReturn(request, 1, 2, measures(0))

    trainer.run(observation(clock), run)
    assert trainer.snapshot()["completed"] == 1


def test_unknown_costs_stop_fit_and_future_observation_is_rejected_before_work():
    _, clock, trainer = setup()
    with pytest.raises(ValueError, match="arrived"):
        trainer.run(observation(LogicalClock(1)), lambda *a: pytest.fail("future data"))
    assert trainer.snapshot()["attempts"] == 0
    with pytest.raises(ValueError, match="execution seconds"):
        trainer.run(
            observation(clock),
            operation(
                clock,
                0,
                result_change=lambda r: replace(
                    r, measures=replace(r.measures, execution_seconds=None)
                ),
            ),
        )
    assert trainer.snapshot()["stopped"]


def test_fitted_controller_requires_complete_original_objective_and_empirical_fit():
    _, _, trainer = trained()
    fitted = trainer.freeze()
    assert type(fitted) is FittedController
    with pytest.raises(ValueError):
        replace(fitted, estimates=fitted.estimates[:3])
