"""Deferred typed captures with scripted factories/evaluators and no native work."""

from copy import deepcopy
import pickle
from typing import Any

import pytest

from src.adapters import minigrid_deferred_probe_rollout as deferred
from src.adapters import minigrid_probe_rollout as live
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from test_minigrid_probe_rollout import Learner, ports as ports, report as live_report


class FrozenReader:
    def __init__(self, work):
        self.work = work
        self.state = None
        work.setdefault("factories", []).append(self)

    def restore_state(self, state):
        self.work.setdefault("restores", []).append((self, deepcopy(state)))
        self.state = deepcopy(state)
        hook = self.work.get("restore_hook")
        if hook is not None:
            hook(self, state)

    def snapshot_state(self):
        assert self.state is not None
        hook = self.work.get("snapshot_hook")
        if hook is not None:
            hook(self)
        return deepcopy(self.state)


def boundary(read_ports=None):
    if read_ports is not None:
        assert read_ports() is None


def completed_budget():
    budget = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=16, max_replay_examples=0), lambda: 1.0
    )
    # Engineering fixture: a literal paid counter, not executed training.
    budget.updates_completed = 16
    return budget


@pytest.fixture
def prepared(monkeypatch, ports):
    work: dict[str, Any] = {}
    monkeypatch.setattr(deferred, "monotonic", lambda: 1.0)
    monkeypatch.setattr(deferred.memory, "learner", lambda: FrozenReader(work))
    manager = deferred.DeferredProbeRollouts()
    budgets = []
    for origin in (0, 128):
        observe = manager.observer(origin)
        for phase, relative in (("after_probe", 16), ("after_return", 32)):
            observe(Learner(origin + relative), phase, boundary)
        pair = (completed_budget(), completed_budget())
        budgets.append(pair)
        manager.bind_completed(origin, *pair)
    return manager, work, budgets, ports


def test_should_capture_all_training_before_two_frozen_readers_and_four_restores(prepared):
    manager, work, budgets, ports = prepared
    initial = manager.report()
    assert not ports["calls"] and work == {}
    assert [tuple((c["source_updates"], c["phase"])) for c in initial["captures"]] == list(
        deferred.ORDER
    )
    assert all(c["status"] == "complete" for c in initial["captures"])
    assert initial["capture_work"]["snapshot_calls_returned"] == 4
    assert initial["capture_work"]["payload_bytes_retained"] <= deferred.MAX_PAYLOAD_BYTES
    assert initial["capture_work"]["logical_payload_bytes_retained"] == 4 * 2424
    before = [
        (b.last_clock, b.updates_completed, b.replay_examples_completed)
        for pair in budgets
        for b in pair
    ]
    result = manager.run(check_boundary=lambda: None)
    assert result["status"] == "complete", result["reason"]
    assert [(r["source_updates"], r["phase"]) for r in result["records"]] == list(deferred.ORDER)
    assert len(work["factories"]) == 2 and len(work["restores"]) == 4
    assert len(ports["calls"]) == 4
    assert [call[1].completed_updates for call in ports["calls"]] == [16, 32, 144, 160]
    assert result["capture_work"]["frozen_reader_factories_returned"] == 2
    assert result["capture_work"]["frozen_reader_restores_returned"] == 4
    assert before == [
        (b.last_clock, b.updates_completed, b.replay_examples_completed)
        for pair in budgets
        for b in pair
    ]
    manager.check()
    result["records"][0]["status"] = "forged"
    assert manager.report()["records"][0]["status"] == "complete"


def test_should_detach_original_endpoint_before_later_probe_updates(monkeypatch):
    monkeypatch.setattr(deferred, "monotonic", lambda: 1.0)
    manager, learner = deferred.DeferredProbeRollouts(), Learner(16)
    before = deepcopy(learner.state)
    manager.observer(0)(learner, "after_probe", boundary)
    learner.state = Learner(32).state
    manager.check()
    payload = manager._captures[0][2]
    assert pickle.loads(payload) == before
    assert manager.report()["captures"][0]["native_updates"] == 16


@pytest.mark.parametrize("origin", [True, 1, 32, 128.0, "0"])
def test_should_refuse_invalid_origin_without_snapshot_work(origin):
    manager = deferred.DeferredProbeRollouts()
    with pytest.raises(ValueError):
        manager.observer(origin)
    assert manager.report()["capture_work"]["snapshot_calls_attempted"] == 0


@pytest.mark.parametrize(
    "fault", ["order", "clock", "post_guard", "no_read", "double_read", "boundary_result"]
)
def test_should_refuse_capture_failure_without_retaining_unqualified_payload(monkeypatch, fault):
    monkeypatch.setattr(deferred, "monotonic", lambda: 1.0)
    manager = deferred.DeferredProbeRollouts()
    learner = Learner(17 if fault == "clock" else 16)
    phase = "after_return" if fault == "order" else "after_probe"

    def qualify(read_ports=None):
        if fault != "no_read" and read_ports is not None:
            read_ports()
            if fault == "double_read":
                read_ports()
        if fault == "post_guard":
            raise ValueError("closing parent proof changed")
        if fault == "boundary_result":
            return 1
        return None

    with pytest.raises(ValueError):
        manager.observer(0)(learner, phase, qualify)
    saved = manager.report()
    assert saved["status"] == "failed"
    assert saved["capture_work"]["payload_bytes_retained"] == 0
    assert manager._captures == ()
    if fault not in ("order", "no_read"):
        assert saved["capture_work"]["snapshot_calls_returned"] == 1


@pytest.mark.parametrize("limit", ["MAX_STATE_BYTES", "MAX_PAYLOAD_BYTES", "MAX_LOGICAL_BYTES"])
def test_should_keep_returned_snapshot_cost_when_a_prospective_storage_limit_refuses(
    monkeypatch, limit
):
    monkeypatch.setattr(deferred, "monotonic", lambda: 1.0)
    monkeypatch.setattr(deferred, limit, 1)
    manager = deferred.DeferredProbeRollouts()
    with pytest.raises(ValueError):
        manager.observer(0)(Learner(), "after_probe", boundary)
    saved = manager.report()
    assert saved["capture_work"]["snapshot_calls_returned"] == 1
    assert saved["capture_work"]["snapshot_copy_calls_returned"] == 1
    assert saved["captures"][0]["payload_bytes"] > 1
    assert saved["capture_work"]["payload_bytes_retained"] == 0


@pytest.mark.parametrize("field", ["_captures", "_bindings", "_proof", "_owner_proof"])
def test_should_refuse_private_registry_or_proof_replacement_before_any_games(prepared, field):
    manager, work, _, ports = prepared
    original = getattr(manager, field)
    replacement = tuple(list(original)) if type(original) is tuple else lambda: None
    setattr(manager, field, replacement)
    with pytest.raises(ValueError):
        manager.check()
    result = manager.run(check_boundary=lambda: None)
    assert result["status"] == "failed" and result["records"] == []
    assert len(result["captures"]) == 4 and result["capture_work"]["snapshot_calls_returned"] == 4
    assert work == {} and not ports["calls"]


@pytest.mark.parametrize("fault", ["updates", "clock", "limits", "replay"])
def test_should_refuse_completed_original_budget_changes_without_observing_new_clocks(
    prepared, fault
):
    manager, work, budgets, ports = prepared
    budget = budgets[0][0]
    if fault == "updates":
        budget.updates_completed = 15
    elif fault == "clock":
        budget.last_clock += 0.25
    elif fault == "limits":
        object.__setattr__(budget.budget, "max_training_updates", 17)
    else:
        budget.replay_examples_completed = 1
    result = manager.run(check_boundary=lambda: None)
    assert result["status"] == "failed" and result["records"] == []
    assert work == {} and not ports["calls"]


def test_should_refuse_a_bad_full_restore_before_admitting_any_game(prepared):
    manager, work, _, ports = prepared

    def wrong_restore(reader, supplied):
        # A same-clock, valid full snapshot with different action counts.
        entry = supplied.native_state.entries[0]
        object.__setattr__(entry, "counts", (0, entry.counts[0], 0, 0, 0, 0, 0))
        reader.state = supplied

    work["restore_hook"] = wrong_restore
    result = manager.run(check_boundary=lambda: None)
    assert result["status"] == "failed" and result["records"] == []
    assert result["capture_work"]["frozen_reader_restores_returned"] == 1
    assert not ports["calls"]


@pytest.mark.parametrize("fault", ["partial", "lost", "closing"])
def test_should_keep_returned_or_unknown_game_work_before_operational_failure(prepared, fault):
    manager, _, _, ports = prepared
    closed = [False]

    def read(owner, state, options):
        if fault == "partial":
            partial = live_report()
            partial.update(status="failed", reason="scripted interruption")
            raise live.MiniGridEvaluationFailed(partial)
        if fault == "lost":
            raise RuntimeError("lost admitted delegate report")
        closed[0] = True
        return live_report()

    def guard():
        if closed[0]:
            raise ValueError("phase closing proof changed")

    ports["callback"] = read
    result = manager.run(check_boundary=guard)
    assert result["status"] == "failed" and len(result["records"]) == 1
    paid = result["records"][0]
    if fault == "lost":
        assert paid["report"] is None and paid["evaluator_admitted"] is True
    else:
        assert paid["report"]["work"]["live_prediction_calls_returned"] == 4
    assert len(ports["calls"]) == 1
    assert manager.report()["status"] == "failed"


def test_should_separate_deferred_capture_costs_from_existing_game_lower_bounds(prepared):
    manager, _, _, _ = prepared
    report = manager.run(check_boundary=lambda: None)
    phase = dict(
        status="complete", reason=None, entry_validated=True, exit_validated=True, report=report
    )
    work: dict[str, Any] = dict(
        total_score_forwards_completed=94,
        total_score_rows_completed=846,
        source_environment_steps_returned=90,
        failure_work_may_be_unobserved=False,
    )
    deferred.add_work(work, phase)
    assert work["learner_live_episodes"] == 16
    assert work["total_prediction_forwards_completed"] == 110
    assert work["deferred_capture"]["frozen_reader_restores_returned"] == 4
    missing = dict(phase, status="failed", report=None, exit_validated=False)
    deferred.add_work(work, missing)
    assert work["learner_live_episodes"] is None
    assert work["total_prediction_rows_completed"] is None
    assert work["failure_work_may_be_unobserved"] is True


@pytest.mark.parametrize(
    "flag,memory,inline,valid",
    [
        (False, False, False, True),
        (True, True, False, True),
        (1, True, False, False),
        (True, False, False, False),
        (True, True, True, False),
        (True, True, 0, False),
    ],
)
def test_should_admit_only_the_explicit_mutually_exclusive_recipe(flag, memory, inline, valid):
    if valid:
        deferred.validate_option(flag, memory, inline)
    else:
        with pytest.raises(ValueError):
            deferred.validate_option(flag, memory, inline)


def test_should_keep_first_failure_and_bound_receipts_when_public_summary_is_replaced(monkeypatch):
    manager = deferred.DeferredProbeRollouts()
    with pytest.raises(ValueError):
        manager.observer(0)(Learner(17), "after_probe", boundary)
    saved = manager._report()
    assert len(saved["captures"]) == 1
    monkeypatch.setattr(manager, "report", lambda: dict(status="capturing"))
    for _ in range(6):
        with pytest.raises(ValueError):
            manager.observer(0)(Learner(16), "after_probe", boundary)
    refused = manager._report()
    assert refused["reason"] == saved["reason"]
    assert refused["captures"] == saved["captures"]
    assert refused["capture_work"]["snapshot_calls_attempted"] == 1
    manager.run(check_boundary=lambda: None)
    assert manager._report()["reason"] == saved["reason"]
