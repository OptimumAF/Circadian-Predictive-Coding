"""Cheap direct helper tests for opt-in diagnostics and original budget isolation."""

from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import src.adapters.key_door_learnability as helper
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget


@dataclass
class State:
    completed_updates: int = 8
    training_examples: int = 64
    parameter: float = 0.5


class Source:
    def __init__(self):
        self.state = State()
        self.snapshots = 0

    def snapshot_state(self):
        self.snapshots += 1
        return deepcopy(self.state)


class Clock:
    def __init__(self):
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return 1.0


def report(*, decisions=False):
    result: dict[str, Any] = {
        "status": "complete",
        "reason": None,
        "completed_updates": 16,
        "checkpoints": [
            {
                "updates": point,
                "scores": {
                    context: {"ce": 2.0 - point / 32, "accuracy": 0.125 + point / 32}
                    for context in ("E-red", "E-blue")
                },
            }
            for point in (0, 4, 8, 16)
        ],
    }
    if decisions:
        result["decision_diagnostics_enabled"] = True
        for point in result["checkpoints"]:
            point["decision_diagnostics"] = {
                context: {"rows": 8} for context in ("E-red", "E-blue")
            }
    return result


@pytest.fixture
def setup(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(helper, "monotonic", clock)
    attachments = []
    monkeypatch.setattr(
        ToyBudgetSession, "attach_memory", lambda self, sampler: attachments.append(sampler)
    )
    wake = ToyBudgetSession(ToyExecutionBudget(max_training_updates=8), clock)
    replay = ToyBudgetSession(ToyExecutionBudget(max_replay_examples=32), clock)
    wake.record_update()
    replay.record_replay(3)
    source = Source()
    return SimpleNamespace(
        clock=clock,
        attachments=attachments,
        wake=wake,
        replay=replay,
        source=source,
        state=State(),
        row={"probes": helper.probe_slots()},
        calls=[],
        budgets=[],
        sampler=object(),
        data=(
            (SimpleNamespace(context="E-red"),),
            tuple(SimpleNamespace(context=c) for c in ("E-red", "E-blue")),
        ),
    )


def runner_for(setup, *, decisions=False, alter=None, failure=None, charges=16):
    def runner(source, state, training, evaluation, **kwargs):
        setup.calls.append(kwargs.copy())
        budget = kwargs["budget"]
        setup.budgets.append(budget)
        for _ in range(charges):
            budget.before_update()
            budget.record_update()
        result = report(decisions=decisions)
        if alter is not None:
            alter(result)
        if failure is not None:
            raise failure
        return result

    return runner


def run(setup, runner, **kwargs):
    return helper.run_checkpoint_probe(
        setup.row,
        setup.source,
        setup.state,
        setup.data,
        checkpoint=8,
        wake=setup.wake,
        replay=setup.replay,
        deadline=100.0,
        sampler=setup.sampler,
        same_state=lambda left, right: left == right,
        probe_runner=runner,
        **kwargs,
    )


def slot(setup):
    return setup.row["probes"][1]


@pytest.mark.parametrize("explicit", [False, True])
def test_should_preserve_default_callback_keyword_shape_and_complete_report(setup, explicit):
    kwargs = {"decision_diagnostics": False} if explicit else {}
    before_wake, before_replay = helper._bindings(setup.wake, setup.replay)
    run(setup, runner_for(setup), **kwargs)
    assert set(setup.calls[0]) == {"budget", "checkpoint_updates"}
    assert setup.calls[0]["checkpoint_updates"] == (0, 4, 8, 16)
    assert slot(setup)["status"] == "complete"
    assert slot(setup)["charged_updates"] == 16
    assert slot(setup)["report"] == report()
    assert slot(setup)["outcomes"]["ce_gain"] == 0.5
    assert helper._bindings(setup.wake, setup.replay) == (before_wake, before_replay)


def test_should_forward_only_true_and_keep_reserved_context_maps_without_recomputation(setup):
    run(setup, runner_for(setup, decisions=True), decision_diagnostics=True)
    assert set(setup.calls[0]) == {"budget", "checkpoint_updates", "decision_diagnostics"}
    assert setup.calls[0]["decision_diagnostics"] is True
    assert slot(setup)["report"] == report(decisions=True)
    assert slot(setup)["source_unchanged"] is True
    assert slot(setup)["state_unchanged"] is True
    assert slot(setup)["budget_binding_preserved"] is True
    assert setup.budgets[0] is not setup.wake and setup.budgets[0] is not setup.replay
    assert (setup.wake.updates_completed, setup.replay.replay_examples_completed) == (1, 3)
    assert setup.attachments == [setup.sampler]


@pytest.mark.parametrize("value", [None, 0, 1, np.bool_(True), "true"])
def test_should_refuse_nonbool_before_slot_clock_source_budget_or_callback(setup, value):
    setup.row = {}  # Flag validation must precede even probe-slot access.
    before = setup.clock.calls
    with pytest.raises(ValueError, match="exact bool"):
        run(setup, runner_for(setup), decision_diagnostics=value)
    assert setup.clock.calls == before and setup.source.snapshots == 0
    assert setup.calls == setup.budgets == setup.attachments == []


@pytest.mark.parametrize("value", [None, False, 1, "true"])
def test_should_refuse_unaffirmed_requested_diagnostics_after_permanent_charges(setup, value):
    def alter(result):
        if value is None:
            result.pop("decision_diagnostics_enabled")
        else:
            result["decision_diagnostics_enabled"] = value

    with pytest.raises(ValueError, match="explicitly enabled"):
        run(setup, runner_for(setup, decisions=True, alter=alter), decision_diagnostics=True)
    assert slot(setup)["status"] == "failed" and slot(setup)["charged_updates"] == 16
    assert slot(setup)["report"]["completed_updates"] == 16
    assert slot(setup)["budget_binding_preserved"] is True
    assert setup.row["probes"][0]["status"] == "not_run"
    assert len(setup.calls) == 1


@pytest.mark.parametrize(
    "mode", ["missing_map", "missing_context", "not_dict", "context_not_dict", "later_point"]
)
def test_should_refuse_missing_reserved_maps_at_every_reported_checkpoint(setup, mode):
    def alter(result):
        point = result["checkpoints"][-1 if mode == "later_point" else 0]
        if mode in ("missing_map", "later_point"):
            point.pop("decision_diagnostics")
        elif mode == "missing_context":
            point["decision_diagnostics"].pop("E-blue")
        elif mode == "not_dict":
            point["decision_diagnostics"] = []
        else:
            point["decision_diagnostics"]["E-blue"] = None

    with pytest.raises(ValueError, match="reserved context maps"):
        run(setup, runner_for(setup, decisions=True, alter=alter), decision_diagnostics=True)
    assert slot(setup)["charged_updates"] == 16 and slot(setup)["status"] == "failed"
    assert slot(setup)["outcomes"] is None and slot(setup)["report"] is not None
    assert slot(setup)["budget_binding_preserved"] is True


def test_should_use_actual_reserved_context_identifiers_instead_of_fixed_color_names(setup):
    setup.data = ((), (SimpleNamespace(context="other-reserved"),))
    with pytest.raises(ValueError, match="reserved context maps"):
        run(setup, runner_for(setup, decisions=True), decision_diagnostics=True)
    assert slot(setup)["charged_updates"] == 16


def test_should_preserve_runner_primary_fault_and_partial_charges_before_diagnostic_validation(
    setup,
):
    error = RuntimeError("probe prediction failed")
    with pytest.raises(RuntimeError) as caught:
        run(setup, runner_for(setup, failure=error, charges=3), decision_diagnostics=True)
    assert caught.value is error
    assert slot(setup)["report"] is None and slot(setup)["charged_updates"] == 3
    assert slot(setup)["budget_binding_preserved"] is True
    assert "probe prediction failed" in slot(setup)["reason"]


def test_should_preserve_incomplete_report_before_requested_metadata_checks(setup):
    def alter(result):
        result.update(status="incomplete", reason="probe budget spent")

    with pytest.raises(RuntimeError, match="probe budget spent"):
        run(setup, runner_for(setup, alter=alter), decision_diagnostics=True)
    assert slot(setup)["charged_updates"] == 16
    assert slot(setup)["report"]["status"] == "incomplete"


@pytest.mark.parametrize("mutated", ["source", "state", "wake", "replay", "probe_budget"])
def test_should_refuse_changed_source_or_original_budget_bindings(setup, mutated):
    def alter(result):
        if mutated == "source":
            setup.source.state.parameter = -1
        elif mutated == "state":
            setup.state.parameter = -1
        elif mutated == "wake":
            setup.wake.budget = ToyExecutionBudget(max_training_updates=9)
        elif mutated == "replay":
            setup.replay.record_replay(1)
        else:
            setup.budgets[0].budget = ToyExecutionBudget(max_training_updates=17)

    with pytest.raises(ValueError, match="changed source/checkpoint"):
        run(setup, runner_for(setup, decisions=True, alter=alter), decision_diagnostics=True)
    assert slot(setup)["status"] == "failed" and slot(setup)["charged_updates"] == 16
    assert slot(setup)["readback_error"] is not None


def test_should_keep_missing_diagnostics_primary_when_isolation_readback_also_fails(setup):
    def alter(result):
        result.pop("decision_diagnostics_enabled")
        setup.source.state.parameter = -1

    with pytest.raises(ValueError, match="explicitly enabled"):
        run(setup, runner_for(setup, decisions=True, alter=alter), decision_diagnostics=True)
    assert "explicitly enabled" in slot(setup)["reason"]
    assert "changed source/checkpoint" in slot(setup)["readback_error"]
    assert slot(setup)["charged_updates"] == 16
