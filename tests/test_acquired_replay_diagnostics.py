"""Replay-aware acquired readouts and the existing320-wake accounting gate."""

from copy import deepcopy
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from src.adapters import acquired_task_return as acquired
from src.adapters import key_door_pilot as shared
from src.adapters.return_replay_accounting import check_replay_progress
from src.app.coupled_attribution import _same_value
from src.app.key_door_policy_evaluation import KeyDoorEvaluationFailed
from test_acquired_task_return import budget
from test_four_phase_replay_accounting import fixture as accounting_fixture, check, NoExternalWork
from test_train_policy_diagnostics import Learner, state


def clocks():
    return [
        dict(
            wake_updates=wake,
            replay_committed_updates=replay,
            replay_charged_examples=8 * replay,
            native_updates=native,
            native_training_examples=rows,
        )
        for wake, replay, native, rows in (
            (192, 96, 288, 2304),
            (256, 128, 384, 3072),
            (320, 160, 480, 3840),
        )
    ]


def states():
    return [(point["wake_updates"], state(point["native_updates"])) for point in clocks()]


@pytest.fixture(autouse=True)
def fake_state_comparison(monkeypatch):
    monkeypatch.setattr(acquired, "_same_state", _same_value)
    monkeypatch.setattr(shared, "_same_state", _same_value)


@pytest.mark.parametrize("control", ["recent48", "coverage"])
def test_should_bind_replay_clock_metadata_and_keep_wake_labeled_twelve_slots(control):
    first, second = (
        acquired.new_diagnostic(replay_control=control),
        acquired.new_diagnostic(replay_control=control),
    )
    assert first["replay_enabled"] is True and first["replay_control"] == control
    assert first["checkpoint_clocks"] == clocks()
    assert [slot["checkpoint_updates"] for slot in first["episodes"]] == [192] * 4 + [256] * 4 + [
        320
    ] * 4
    first["checkpoint_clocks"][0]["native_updates"] = 192
    assert second["checkpoint_clocks"] == clocks()


def test_should_preserve_default_schema_and_explicit_none_without_replay_metadata():
    implicit, explicit = acquired.new_diagnostic(), acquired.new_diagnostic(replay_control=None)
    assert implicit == explicit and implicit["replay_enabled"] is False
    assert "replay_control" not in implicit and "checkpoint_clocks" not in implicit
    reports = []
    for options in ({}, {"replay_control": None}):
        report = acquired.new_diagnostic(**options)
        acquired.evaluate_acquired_tasks(
            report,
            Learner(),
            [(p, state(p)) for p in (192, 256, 320)],
            budget(),
            check_boundary=lambda: None,
            **options,
        )
        reports.append(report)
    assert reports[0] == reports[1]


class StringSubclass(str):
    pass


@pytest.mark.parametrize(
    "control",
    [
        True,
        False,
        0,
        1,
        "",
        "off",
        "recent16",
        "recent48 ",
        "Coverage",
        np.str_("coverage"),
        StringSubclass("coverage"),
    ],
)
@pytest.mark.parametrize("entry", ["allocate", "evaluate"])
def test_should_refuse_unfrozen_controls_before_source_callbacks_or_budget_access(control, entry):
    source, calls = Learner(), []
    with pytest.raises(ValueError, match="replay_control must"):
        if entry == "allocate":
            acquired.new_diagnostic(replay_control=control)
        else:
            blocked: Any = object()
            acquired.evaluate_acquired_tasks(
                {},
                source,
                [],
                blocked,
                check_boundary=lambda: calls.append(1),
                replay_control=control,
            )
    assert source.trace == [] and calls == []


@pytest.mark.parametrize("control", ["recent48", "coverage"])
def test_should_read_owned_replay_native_states_but_report_all_wake_outcomes(control):
    observed = []

    def observe(event, actor):
        if event == "predict":
            observed.append((actor.state.completed_updates, actor.state.training_examples))

    source, checkpoints, session = Learner(hook=observe), states(), budget()
    before = deepcopy((source.state, checkpoints))
    report = acquired.new_diagnostic(replay_control=control)
    acquired.evaluate_acquired_tasks(
        report, source, checkpoints, session, check_boundary=lambda: None, replay_control=control
    )
    assert report["status"] == "complete" and session.updates_completed == 320
    assert _same_value((source.state, checkpoints), before)
    assert observed == [(288, 2304)] * 48 + [(384, 3072)] * 48 + [(480, 3840)] * 48
    assert [point["updates"] for point in report["outcomes"]] == [192, 256, 320]
    assert all(slot["identity"]["replay_enabled"] is True for slot in report["episodes"])
    assert (
        report["work"]["live_sessions_started"] == 12
        and report["work"]["live_callback_returns"] == 144
    )
    assert report["initial_A_acquired"] is False and "inconclusive" in report["interpretation"]


@pytest.mark.parametrize(
    "fault", ["wake_native", "wrong_native", "rows", "control", "clocks", "wake_budget"]
)
def test_should_refuse_wrong_replay_plan_or_native_progress_before_external_work(fault):
    report, checkpoints, session, source = (
        acquired.new_diagnostic(replay_control="coverage"),
        states(),
        budget(),
        Learner(),
    )
    if fault == "wake_native":
        checkpoints[-1] = (320, state(320))
    elif fault == "wrong_native":
        checkpoints[-1][1].completed_updates = 479
    elif fault == "rows":
        checkpoints[-1][1].training_examples = 3839
    elif fault == "control":
        report["replay_control"] = "recent48"
    elif fault == "clocks":
        report["checkpoint_clocks"][0]["native_updates"] = 192
    else:
        session = budget(480, 480)
    with pytest.raises(ValueError):
        acquired.evaluate_acquired_tasks(
            report,
            source,
            checkpoints,
            session,
            check_boundary=lambda: None,
            replay_control="coverage",
        )
    assert source.trace == []


@pytest.mark.parametrize(
    "fault",
    [
        "control",
        "enabled",
        "clock_value",
        "clock_container",
        "clock_record",
        "native_state",
        "preceding",
    ],
)
def test_should_stop_on_replaced_replay_clock_or_preceding_evidence_and_keep_charges(fault):
    report, checkpoints, session = (
        acquired.new_diagnostic(replay_control="coverage"),
        states(),
        budget(),
    )
    changed = False

    def corrupt(event, actor):
        nonlocal changed
        if event == "predict" and not changed:
            changed = True
            if fault == "control":
                report["replay_control"] = "recent48"
            elif fault == "enabled":
                report["replay_enabled"] = False
            elif fault == "clock_value":
                report["checkpoint_clocks"][0]["native_updates"] = 289
            elif fault == "clock_container":
                report["checkpoint_clocks"] = deepcopy(report["checkpoint_clocks"])
            elif fault == "clock_record":
                report["checkpoint_clocks"][0] = deepcopy(report["checkpoint_clocks"][0])
            elif fault == "native_state":
                checkpoints[-1][1].completed_updates = 481

    source = Learner(hook=corrupt)

    def guard():
        if fault == "preceding" and report["episodes"][0]["state_unchanged"] is True:
            report["episodes"][0]["return"] = 7.0

    expected = ValueError if fault == "preceding" else KeyDoorEvaluationFailed
    with pytest.raises(expected, match="changed"):
        acquired.evaluate_acquired_tasks(
            report, source, checkpoints, session, check_boundary=guard, replay_control="coverage"
        )
    assert report["status"] == "failed" and session.updates_completed == 320
    assert source.trace.count("session") < 12 and report["metrics"] is None


@pytest.mark.parametrize("count", [0, 1, 2, 192, 256, 319, 320])
def test_should_account_for320_wakes_with160_existing_events_and1280_exposures(count):
    data = accounting_fixture(count, 320)
    before = deepcopy((data.helper.saved, data.sources, vars(data.state)))
    result = check(data, total_updates=320)
    assert result == dict(
        wake_updates=count,
        replay_committed_updates=count // 2,
        replay_charged_examples=8 * (count // 2),
        native_updates=count + count // 2,
        native_training_examples=8 * (count + count // 2),
    )
    assert len(data.helper.saved["events"]) == 160
    assert data.budget.budget.max_replay_examples == 1280
    assert data.budget.replay_examples_completed == 8 * (count // 2)
    assert (data.helper.saved, data.sources, vars(data.state)) == before


@pytest.mark.parametrize("total", [192, 256])
def test_should_preserve_existing_total_update_accounting(total):
    data = accounting_fixture(total, total)
    options = {} if total == 192 else {"total_updates": total}
    assert check(data, **options)["native_updates"] == total + total // 2
    assert len(data.helper.saved["events"]) == total // 2


@pytest.mark.parametrize("value", [True, 320.0, np.int64(320), None, "320", 319, 321, 480])
def test_should_reject_invalid320_accounting_total_before_external_work(value):
    blocked = NoExternalWork()
    with pytest.raises(ValueError, match="native integer"):
        check_replay_progress(
            blocked, blocked, 0, blocked, training_sources={}, total_updates=value
        )


@pytest.mark.parametrize("fault", ["receipt", "source", "charge", "native", "partial", "future"])
def test_should_preserve320_replay_commit_source_and_partial_failure_refusals(fault):
    data = accounting_fixture(256 if fault == "future" else 320, 320)
    if fault == "receipt":
        first, *rest = data.helper._runtime.applied_consolidations
        data.helper._runtime.applied_consolidations = (replace(first, event_id="replaced"), *rest)
    elif fault == "source":
        data.helper.saved["events"][0]["selected_episodes"][0]["payload"] = [999]
    elif fault == "charge":
        data.budget.replay_examples_completed -= 8
    elif fault == "native":
        data.state.completed_updates = 479
    elif fault == "partial":
        data.helper.saved["events"][0]["committed_updates"] = 0
    else:
        data.helper.saved["events"][128]["attempted_updates"] = 1
    charge = data.budget.replay_examples_completed
    with pytest.raises(ValueError):
        check(data, total_updates=320)
    assert data.budget.replay_examples_completed == charge
