"""Qualified deferred composition with scripted state/read ports and no physics."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from src.adapters import minigrid_deferred_probe_rollout as deferred
from src.adapters import minigrid_learnability as pilot
from test_minigrid_deferred_games_integration import phase_budget
from test_minigrid_deferred_probe_rollout import prepared as prepared
from test_minigrid_probe_rollout import ports as ports, report as game_report


def run_phase(manager, *, resources=lambda: None, check_phase=lambda: None):
    budget, _ = phase_budget()
    result = pilot._run_deferred_phase(
        manager, [dict(budget=budget)], check_phase=check_phase, resources=resources
    )
    return result, budget


def test_should_complete_all_four_qualified_endpoints_with_original_paid_budgets(prepared):
    manager, work, budgets, ports = prepared
    before = deepcopy(manager.report())
    bindings = [
        (budget.last_clock, budget.updates_completed, budget.replay_examples_completed)
        for pair in budgets
        for budget in pair
    ]
    result, benchmark = run_phase(manager)
    assert result["status"] == "complete", result["reason"]
    assert result["entry_validated"] is result["exit_validated"] is True
    saved = result["report"]
    assert saved["captures"] == before["captures"]
    assert [(row["source_updates"], row["phase"]) for row in saved["records"]] == list(
        deferred.ORDER
    )
    assert all(row["status"] == "complete" for row in saved["records"])
    assert len(ports["calls"]) == len(work["restores"]) == 4
    assert len(work["factories"]) == 2
    assert bindings == [
        (budget.last_clock, budget.updates_completed, budget.replay_examples_completed)
        for pair in budgets
        for budget in pair
    ]
    assert benchmark.updates_completed == 128 and benchmark.replay_examples_completed == 0
    saved["records"][0]["report"]["status"] = "forged"
    assert manager.report()["records"][0]["report"]["status"] == "complete"


def test_should_preserve_no_argument_legacy_callback_and_four_paid_records(prepared):
    manager, _, _, ports = prepared
    calls = []

    def legacy():
        calls.append("legacy")

    saved = manager.run(check_boundary=legacy)
    assert saved["status"] == "complete", saved["reason"]
    assert calls and len(saved["records"]) == len(ports["calls"]) == 4


@pytest.mark.parametrize("field", ["_captures", "_bindings", "_proof", "_owner_proof"])
def test_should_refuse_resource_port_registry_replacement_before_factory_or_reads(prepared, field):
    manager, work, _, ports = prepared
    resources = []

    def corrupt():
        resources.append("resource")
        original = getattr(manager, field)
        replacement = tuple(list(original)) if type(original) is tuple else lambda: None
        setattr(manager, field, replacement)

    result, _ = run_phase(manager, resources=corrupt)
    assert result["status"] == "failed" and result["entry_validated"] is False
    assert result["reason"]
    assert resources == ["resource"]
    assert not ports["calls"] and work == {}
    assert len(result["report"]["captures"]) == 4


def test_should_share_the_adjoining_proof_between_resource_and_supplied_read_groups():
    events: list[str] = []

    def games(*, check_boundary):
        events.clear()  # Exclude phase-entry validation from this port-group count.
        check_boundary(lambda: events.append("read"))
        # No external callback intervenes between closing the resource group
        # and opening the reads, so that one proof protects both boundaries.
        assert events == ["proof", "resource", "proof", "read", "proof", "resource", "proof"]
        events.clear()
        check_boundary()
        assert events == ["proof", "resource", "proof"]
        return dict(status="complete", reason=None)

    manager = SimpleNamespace(
        check=lambda: events.append("proof"),
        run_qualified=games,
        report=lambda: None,
    )
    result, _ = run_phase(manager, resources=lambda: events.append("resource"))
    assert result["status"] == "complete", result["reason"]


def test_should_observe_closing_resources_after_failed_read_and_keep_the_original_error():
    events = []
    read_failed = [False]

    def resources():
        events.append("resource")
        if read_failed[0]:
            raise RuntimeError("later resource observation failure")

    def read():
        events.append("read")
        read_failed[0] = True
        raise OSError("original supplied read failure")

    def games(*, check_boundary):
        events.clear()
        try:
            check_boundary(read)
        except OSError:
            assert events == ["proof", "resource", "proof", "read", "proof", "resource", "proof"]
            raise
        raise AssertionError("a failed read admitted another operation")

    manager = SimpleNamespace(
        check=lambda: events.append("proof"),
        run_qualified=games,
        report=lambda: dict(status="failed", reason="supplied read interrupted"),
    )
    result, _ = run_phase(manager, resources=resources)
    assert result["status"] == "failed"
    assert "original supplied read failure" in result["reason"]


def test_should_keep_paid_game_when_postread_resources_refuse_and_stop_before_next_game(prepared):
    manager, work, _, ports = prepared
    events = []
    paid_returned = [False]
    snapshot_returned = [False]

    def snapshot(reader):
        if paid_returned[0]:
            events.append("paid snapshot")
            snapshot_returned[0] = True

    def evaluate(reader, state, options):
        paid_returned[0] = True
        return game_report()

    def resources():
        if snapshot_returned[0]:
            events.append("post-read resource refusal")
            raise OSError("post-read resource observation failed")

    work["snapshot_hook"] = snapshot
    ports["callback"] = evaluate
    result, _ = run_phase(manager, resources=resources)
    assert result["status"] == "failed"
    assert "post-read resource observation failed" in result["reason"]
    assert events[:2] == ["paid snapshot", "post-read resource refusal"]
    assert len(ports["calls"]) == len(work["restores"]) == 1
    records = result["report"]["records"]
    assert len(records) == 1 and records[0]["evaluator_admitted"] is True
    assert records[0]["report"]["work"]["live_prediction_calls_returned"] == 4


@pytest.mark.parametrize("port", ["resource", "read"])
def test_should_run_closing_proof_and_keep_first_port_error_when_closing_also_fails(port):
    events = []
    armed = [False]

    def proof():
        events.append("proof")
        if armed[0]:
            raise ValueError("later closing proof failure")

    def fail():
        events.append(port)
        armed[0] = True
        raise OSError("original port failure")

    def games(*, check_boundary):
        check_boundary(fail)
        raise AssertionError("a failed read admitted later work")

    manager = SimpleNamespace(
        check=proof,
        run_qualified=games,
        report=lambda: dict(status="failed", reason="paid phase stopped", records=[]),
    )
    resources = fail if port == "resource" else lambda: events.append("resource")
    result, _ = run_phase(manager, resources=resources)
    assert result["status"] == "failed"
    assert "original port failure" in result["reason"]
    assert events[events.index(port) + 1] == "proof"
    assert events.count(port) == 1
    if port == "read":
        # Phase entry and the read opening resource ran; corruption blocks
        # both the read closing resource and phase-exit resource observation.
        assert events.count("resource") == 2


@pytest.mark.parametrize("port", ["resource", "read"])
@pytest.mark.parametrize("fault", ["counter_type", "clock", "rss", "limits"])
def test_should_refuse_unobserved_budget_changes_in_actual_qualified_port_groups(port, fault):
    budget, _ = phase_budget()
    events = []
    armed = [False]

    def corrupt():
        armed[0] = True
        events.append("corrupt")
        if fault == "counter_type":
            budget.replay_examples_completed = False
        elif fault == "clock":
            budget.last_clock += 0.25
        elif fault == "rss":
            budget.process_rss_segment = replace(budget.process_rss_segment, sample_count=999)
        else:
            object.__setattr__(budget.budget, "max_training_updates", 129)

    def resources():
        if port == "resource" and not armed[0]:
            corrupt()

    def games(*, check_boundary):
        check_boundary(corrupt)
        events.append("forbidden later work")
        return dict(status="complete", reason=None)

    manager = SimpleNamespace(
        check=lambda: events.append("proof"),
        run_qualified=games,
        report=lambda: dict(status="failed", reason="port changed evidence"),
    )
    result = pilot._run_deferred_phase(
        manager,
        [dict(budget=budget)],
        check_phase=lambda: None,
        resources=resources,
    )
    assert result["status"] == "failed" and result["exit_validated"] is False
    assert result["entry_validated"] is (port == "read")
    assert "forbidden later work" not in events
    assert events[events.index("corrupt") + 1] == "proof"
    assert events.count("corrupt") == 1


@pytest.mark.parametrize("failure", ["partial", "lost", "paid_replaced"])
def test_should_stop_after_first_native_failure_and_retain_paid_or_unknown_work(prepared, failure):
    manager, _, _, ports = prepared

    def evaluate(reader, state, options):
        if failure == "partial":
            paid = game_report()
            paid.update(status="failed", reason="original delegated partial failure")
            from src.adapters.minigrid_probe_rollout import MiniGridEvaluationFailed

            raise MiniGridEvaluationFailed(paid)
        if failure == "lost":
            raise OSError("native read lost its report")
        return game_report()

    ports["callback"] = evaluate
    if failure == "paid_replaced":
        # The second endpoint's parent proof must refuse a same-value replacement
        # of the first returned receipt before admitting another native game.
        def corrupt_on_second_restore(reader, supplied):
            if len(ports["calls"]) == 1:
                record = manager._readers[0].records[0]
                record["report"] = deepcopy(record["report"])

        prepared[1]["restore_hook"] = corrupt_on_second_restore
    result, _ = run_phase(manager)
    assert result["status"] == "failed" and len(ports["calls"]) == 1
    records = result["report"]["records"]
    assert len(records) == 1
    paid = records[0]
    if failure == "lost":
        assert paid["evaluator_admitted"] is True and paid["report"] is None
    else:
        assert paid["report"]["work"]["live_prediction_calls_returned"] == 4
    previous = deepcopy(result["report"])
    again = manager.run_qualified(check_boundary=lambda read_ports=None: None)
    assert again["reason"] == previous["reason"]
    assert len(ports["calls"]) == 1
