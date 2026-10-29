"""Opt-in catalog fitting and original counterfactual/shared-runtime integration."""

from copy import deepcopy
from dataclasses import asdict, replace
import json
from typing import Any, cast

import pytest

from src.app.counterfactual_controller import (
    CounterfactualController,
    CatalogCounterfactualRequest,
    CatalogCounterfactualReturn,
)
from src.core.catalog_contextual_controller import (
    ContextualArmCatalog,
    CatalogEstimate,
    CatalogTrialWork,
    FittedCatalogController,
    fit_catalog_choices,
    choose_catalog_contextual,
)
from src.core.adaptation_action import AdaptationAction, AdaptationEnvelope
from src.core.actor_ports import VersionedState
from src.core.controller_objective import ControllerObjective, ControllerMeasures
from src.core.experience import LogicalClock
from test_contextual_controller import observation
from test_categorical_replay_action import setup as replay_setup, ready


def catalog(**changes):
    envelope = AdaptationEnvelope(
        (
            AdaptationAction("wait"),
            AdaptationAction("wake1", 1),
            AdaptationAction("replay_wake", 1, 3),
        ),
        max_decisions=6,
        max_wake_updates=4,
        max_replay_examples=6,
    )
    return ContextualArmCatalog(replace(envelope, **changes))


def estimates(values=(0.0, 0.0, 0.0)):
    return tuple(
        CatalogEstimate(context, arm, value)
        for context in ("clean", "mistakes")
        for arm, value in zip(catalog().arms, values, strict=True)
    )


def test_should_fit_all_six_catalog_cells_and_keep_wait_first_then_declared_order_on_ties():
    assert fit_catalog_choices(catalog(), estimates()) == ("wait", "wait")
    assert fit_catalog_choices(catalog(), estimates((0.0, 1.0, 1.0))) == ("wake1", "wake1")
    assert fit_catalog_choices(catalog(), estimates((-2.0, -1.0, 0.0))) == (
        "replay_wake",
        "replay_wake",
    )
    assert fit_catalog_choices(catalog(), estimates((-1.0, -2.0, -3.0))) == ("wait", "wait")


@pytest.mark.parametrize("fault", ["missing", "order", "wake", "replay", "rate", "gate", "sleep"])
def test_should_refuse_unsupported_or_malformed_catalogs(fault):
    actions = list(catalog().envelope.actions)
    if fault == "missing":
        actions.pop()
    elif fault == "order":
        actions[:2] = reversed(actions[:2])
    elif fault == "wake":
        actions[2] = replace(actions[2], wake_updates=2)
    elif fault == "replay":
        actions[2] = replace(actions[2], replay_examples=0)
    elif fault == "rate":
        actions[2] = replace(actions[2], learning_rate_multiplier=0.5)
    elif fault == "gate":
        actions[2] = replace(actions[2], gate_setting="neutral")
    else:
        actions[2] = replace(actions[2], sleep_updates=1)
    with pytest.raises(ValueError):
        catalog(actions=tuple(actions))


@pytest.mark.parametrize("fault", ["missing", "duplicate", "foreign", "nonfinite", "samples"])
def test_should_require_all_six_exact_finite_catalog_estimates(fault):
    rows = estimates()
    with pytest.raises(ValueError):
        if fault == "missing":
            rows = rows[:-1]
        elif fault == "duplicate":
            rows = rows[:-1] + (rows[0],)
        elif fault == "foreign":
            rows = rows[:-1] + (replace(rows[-1], action_id="unknown"),)
        elif fault == "nonfinite":
            rows = rows[:-1] + (replace(rows[-1], mean_utility=float("nan")),)
        else:
            rows = rows[:-1] + (replace(rows[-1], samples=2),)
        fit_catalog_choices(catalog(), rows)


def policy():
    c = catalog()
    return FittedCatalogController(
        "original",
        "digest",
        ControllerObjective(),
        0,
        estimates((0.0, 0.0, 1.0)),
        ("replay_wake", "replay_wake"),
        catalog=c,
    )


@pytest.mark.parametrize("fault", ["no_feedback", "not_ready", "no_wake", "no_replay"])
def test_should_wait_without_replacing_the_selected_dose_when_credit_or_arrival_is_absent(fault):
    p, o = policy(), observation(LogicalClock())
    credit = 6
    if fault == "no_feedback":
        o = replace(o, feedback=None)
    elif fault == "not_ready":
        o = replace(o, ready_updates=0)
    elif fault == "no_wake":
        o = replace(o, remaining_wake=0)
    else:
        credit = 2
    d = choose_catalog_contextual(
        p, o, envelope=p.catalog.envelope, remaining_replay_examples=credit
    )
    assert d.action_id == "wait"


@pytest.mark.parametrize("fault", ["version", "envelope", "credit", "choice"])
def test_should_refuse_stale_policy_changed_envelope_or_inconsistent_fit(fault):
    p, o = policy(), observation(LogicalClock())
    envelope, credit = p.catalog.envelope, 6
    if fault == "version":
        o = observation(LogicalClock(), version="new")
    elif fault == "envelope":
        envelope = replace(envelope, max_decisions=7)
    elif fault == "credit":
        credit = 7
    else:
        object.__setattr__(p, "choices", ("wake1", "wait"))
    with pytest.raises(ValueError):
        choose_catalog_contextual(p, o, envelope=envelope, remaining_replay_examples=credit)


def setup_catalog():
    source = [VersionedState("original", {"weights": [1], "optimizer": [2], "clock": 96})]
    clock = LogicalClock()
    c = catalog()
    owner = CounterfactualController(
        lambda: deepcopy(source[0]), json.dumps, ControllerObjective(), clock, catalog=c
    )
    return source, clock, c, owner


def measured(gain=0):
    return ControllerMeasures(0, gain, 1, 1, 1, 1, 0.01, 1024, 0, True)


def empty_work():
    return CatalogTrialWork(0, 0, 0, 0, 0, 0, 0)


def operation(clock, gain=0):
    def run(request, state):
        assert type(request) is CatalogCounterfactualRequest
        state["optimizer"][0] = 999
        clock.advance_to(request.started_at + 2)
        return CatalogCounterfactualReturn(
            request, request.started_at + 1, clock.now(), measured(gain), empty_work()
        )

    return run


def trained_catalog():
    source, clock, c, owner = setup_catalog()
    for context in ("clean", "mistakes"):
        for index in range(3):
            gain = int(context == "mistakes" and index == 2)
            owner.run(observation(clock, context), operation(clock, gain))
    return source, clock, c, owner


def test_should_freeze_six_discarded_trials_keep_costs_and_allow_only_catalog_selection():
    source, clock, c, owner = trained_catalog()
    fitted = owner.freeze()
    assert type(fitted) is FittedCatalogController and fitted.choices == ("wait", "replay_wake")
    assert source[0].state["optimizer"] == [2]
    view = owner.application_snapshot()
    assert view["fit"]["attempts"] == view["fit"]["discarded_trials"] == 6
    assert len(view["fit"]["returns"]) == 6
    assert all(r["work"] == asdict(empty_work()) for r in view["fit"]["returns"])
    assert (
        owner.choose(observation(clock), envelope=c.envelope, remaining_replay_examples=6).action_id
        == "replay_wake"
    )
    assert (
        owner.choose(
            observation(clock, "clean"), envelope=c.envelope, remaining_replay_examples=6
        ).action_id
        == "wait"
    )
    with pytest.raises(ValueError, match="already fitted"):
        owner.freeze()
    with pytest.raises(ValueError, match="closed"):
        owner.run(observation(clock), lambda *args: pytest.fail("seventh trial"))


@pytest.mark.parametrize(
    "fault", ["source", "copy", "role", "catalog", "future", "unknown", "cost"]
)
def test_should_stop_paid_invalid_trials_before_reading_poison_metrics(fault):
    source, clock, _, owner = setup_catalog()

    class Poison:
        def __getattribute__(self, key):
            pytest.fail("read invalid trial metrics")

    def run(request, state):
        assert type(request) is CatalogCounterfactualRequest
        clock.advance_to(2)
        if fault == "source":
            source[0].state["weights"][0] += 1
        if fault == "role":
            object.__setattr__(request, "role", "final_test")
        if fault == "catalog":
            object.__setattr__(request, "catalog", catalog(max_decisions=7))
        work = empty_work()
        if fault == "unknown":
            work = CatalogTrialWork(1, 0, 2, 0, 0, 0, 0, False)
        if fault == "cost":
            work = CatalogTrialWork(1, 1, 3, 3, 3, 0, 0)
        return CatalogCounterfactualReturn(
            deepcopy(request) if fault == "copy" else request,
            3 if fault == "future" else 1,
            2,
            cast(Any, Poison()),
            work,
        )

    with pytest.raises(ValueError):
        owner.run(observation(clock), run)
    assert owner.snapshot()["attempts"] == owner.snapshot()["discarded_trials"] == 1
    assert owner.snapshot()["completed"] == 0 and owner.snapshot()["stopped"]
    if fault in ("unknown", "cost"):
        assert owner.snapshot()["failed_work"] is not None
    with pytest.raises(ValueError):
        owner.run(observation(clock), lambda *args: pytest.fail("retry paid failure"))


@pytest.mark.parametrize("fault", ["row", "work", "request", "envelope", "measure"])
def test_should_refuse_corrupted_or_replaced_previous_success_before_another_trial(fault):
    _, clock, _, owner = setup_catalog()
    owner.run(observation(clock), operation(clock))
    returned = cast(CatalogCounterfactualReturn, owner._returns[0])
    if fault == "row":
        owner._rows = (deepcopy(owner._rows[0]),)
    elif fault == "work":
        object.__setattr__(returned, "work", deepcopy(returned.work))
    elif fault == "request":
        object.__setattr__(returned, "request", deepcopy(returned.request))
    elif fault == "envelope":
        object.__setattr__(
            returned.request.catalog, "envelope", deepcopy(returned.request.catalog.envelope)
        )
    else:
        object.__setattr__(returned.measures, "execution_seconds", 2.0)
    with pytest.raises(ValueError, match="preceding catalog trial"):
        owner.run(
            observation(clock), lambda *args: pytest.fail("opened new trial after corruption")
        )
    assert owner.snapshot()["attempts"] == 1


@pytest.mark.parametrize("fault", ["catalog", "fit", "work", "clock", "source"])
def test_should_refuse_changed_frozen_authority_before_catalog_application(fault):
    source, clock, c, owner = trained_catalog()
    owner.freeze()
    if fault == "catalog":
        owner._catalog = deepcopy(owner._catalog)
    elif fault == "fit":
        owner._fitted = deepcopy(owner._fitted)
    elif fault == "work":
        object.__setattr__(owner._returns[0], "work", empty_work())
    elif fault == "clock":
        owner._clock = LogicalClock(clock.now())
    else:
        source[0].state["optimizer"][0] += 1
    with pytest.raises(ValueError):
        owner.choose(observation(clock), envelope=c.envelope, remaining_replay_examples=6)


def test_should_preserve_legacy_default_dataclass_and_snapshot_shapes():
    from test_contextual_controller import trained
    from src.core.contextual_controller import FittedController
    from src.app.counterfactual_controller import CounterfactualRequest

    _, _, owner = trained()
    p = owner.freeze()
    assert type(p) is FittedController
    assert set(asdict(p)) == {
        "learner_version",
        "state_digest",
        "objective",
        "fitted_at",
        "estimates",
        "choices",
    }
    assert set(owner.snapshot()) == {
        "attempts",
        "completed",
        "discarded_trials",
        "stopped",
        "rows",
        "fitted",
        "clock",
    }
    assert set(asdict(CounterfactualRequest("v", "d", "clean", "wait", 0))) == {
        "learner_version",
        "state_digest",
        "context",
        "action_id",
        "started_at",
        "role",
    }
    assert len(owner.application_snapshot()["fit"]["returns"]) == 4


def test_should_fit_six_actual_scalar_trials_then_execute_the_frozen_composite_choice():
    c = catalog()
    shared, runtime, _, source, app_clock, wake, bank, _, _ = replay_setup(
        action_envelope=c.envelope
    )
    original = source.snapshot_state()
    clock = LogicalClock()
    digest = lambda state: json.dumps(asdict(state), sort_keys=True)
    owner = CounterfactualController(
        runtime.actor.snapshot_state, digest, ControllerObjective(), clock, catalog=c
    )
    for context in ("clean", "mistakes"):
        for arm in c.arms:

            def run(request, state):
                assert type(request) is CatalogCounterfactualRequest and request.action_id == arm
                private, r, replay, learner, tick, b_wake, b_replay, _, _ = replay_setup(
                    action_envelope=c.envelope
                )
                assert learner.snapshot_state() == state
                ready(r, tick)
                poll = private.train_action(request.action_id)
                updates, rows = len(learner.calls), sum(learner.calls)
                work = CatalogTrialWork(
                    updates, updates, rows, rows, b_replay.replay_examples_completed, 0, 0
                )
                gain = int(context == "mistakes" and request.action_id == "replay_wake")
                clock.advance_to(request.started_at + 2)
                assert learner.snapshot_state() == r.actor.snapshot_state().state == state
                assert not replay.stopped
                assert b_wake.updates_completed == len(poll.training.updates)
                return CatalogCounterfactualReturn(
                    request, request.started_at + 1, clock.now(), measured(gain), work
                )

            owner.run(observation(clock, context, version="a0"), run)
    owner.freeze()
    costs = owner.application_snapshot()["fit"]["returns"]
    assert sum(x["work"]["attempted_updates"] for x in costs) == 6
    assert sum(x["work"]["attempted_rows"] for x in costs) == 14
    assert sum(x["work"]["replay_examples_charged"] for x in costs) == 6
    ready(runtime, app_clock)
    decision = owner.choose(
        observation(clock, version="a0"),
        envelope=shared.adaptation_snapshot().envelope,
        remaining_replay_examples=bank.remaining_replay_examples,
    )
    assert decision.action_id == "replay_wake"
    result = shared.train_action(decision.action_id)
    assert source.calls == [3, 2] and result.replay_receipt is not None
    assert len(result.training.updates) == wake.updates_completed == 1
    assert bank.replay_examples_completed == 3
    assert source.snapshot_state() == runtime.actor.snapshot_state().state == original


@pytest.mark.parametrize(
    "change",
    [
        {"work_complete": False},
        {"returned_updates": 0},
        {"returned_rows": 4},
        {"replay_examples_charged": 2},
        {"prediction_calls": 0},
        {"attempted_updates": True},
        {"attempted_rows": None},
        {"prediction_rows": -1},
        {"replay_examples_charged": 9},
    ],
)
def test_should_refuse_unknown_partial_or_inconsistent_catalog_work(change):
    with pytest.raises(ValueError):
        work = replace(CatalogTrialWork(2, 2, 5, 5, 3, 1, 2), **change)
        work.require_complete(catalog(), "replay_wake")


def test_should_wait_when_all_literal_utilities_are_negative_and_keep_discarded_costs():
    _, clock, c, owner = setup_catalog()
    for context in ("clean", "mistakes"):
        for index in range(3):

            def run(request, state):
                assert type(request) is CatalogCounterfactualRequest
                clock.advance_to(request.started_at + 2)
                cost = replace(measured(), execution_seconds=float(index + 1))
                return CatalogCounterfactualReturn(
                    request, request.started_at + 1, clock.now(), cost, empty_work()
                )

            owner.run(observation(clock, context), run)
    assert owner.freeze().choices == ("wait", "wait")
    assert (
        sum(
            r["measures"]["execution_seconds"]
            for r in owner.application_snapshot()["fit"]["returns"]
        )
        == 12.0
    )
    assert (
        owner.choose(observation(clock), envelope=c.envelope, remaining_replay_examples=6).action_id
        == "wait"
    )


def test_should_refuse_partial_catalog_and_reentrancy_without_spending_another_trial():
    _, clock, _, owner = setup_catalog()
    with pytest.raises(ValueError, match="six complete"):
        owner.freeze()

    def run(request, state):
        with pytest.raises(ValueError, match="busy"):
            owner.freeze()
        with pytest.raises(ValueError, match="busy"):
            owner.run(observation(clock), lambda *args: pytest.fail("nested trial"))
        assert type(request) is CatalogCounterfactualRequest
        clock.advance_to(2)
        return CatalogCounterfactualReturn(request, 1, 2, measured(), empty_work())

    owner.run(observation(clock), run)
    assert owner.snapshot()["attempts"] == owner.snapshot()["completed"] == 1
    with pytest.raises(ValueError, match="six complete"):
        owner.freeze()


@pytest.mark.parametrize("fault", ["future", "stale", "nontrain"])
def test_should_refuse_future_stale_or_nontrain_observations_before_trials(fault):
    _, clock, _, owner = setup_catalog()
    with pytest.raises(ValueError):
        if fault == "future":
            o = observation(LogicalClock(1))
        elif fault == "stale":
            o = observation(clock, version="new")
        else:
            o = observation(clock)
            object.__setattr__(o.feedback, "role", "final_test")
        owner.run(o, lambda *args: pytest.fail("invalid source callback"))
    assert owner.snapshot()["attempts"] == 0


@pytest.mark.parametrize("value", [None, {}, True, "wake1"])
def test_should_refuse_non_observations_at_the_owner_selection_boundary(value):
    _, _, c, owner = trained_catalog()
    owner.freeze()
    with pytest.raises(ValueError, match="exact causal observation"):
        owner.choose(value, envelope=c.envelope, remaining_replay_examples=6)


def test_should_refuse_utility_overflow_as_invalid_input():
    with pytest.raises(ValueError, match="finite utility"):
        CatalogEstimate("clean", "wait", 2**2048)


def test_should_explain_six_trial_requirement_before_catalog_application():
    _, _, _, owner = setup_catalog()
    with pytest.raises(ValueError, match="six completed catalog trials"):
        owner.application_snapshot()
