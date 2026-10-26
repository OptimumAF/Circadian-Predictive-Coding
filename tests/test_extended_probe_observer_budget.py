"""Explicit H32 readout limits with fake endpoints, clocks and paid reports.

These cases execute no models, training or physics. The fake snapshots are not
evidence of native endpoint validity or learning.
"""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from src.adapters import minigrid_probe_rollout as live
from src.app.toy_execution_budget import ToyExecutionStopped


METHODS = ("bp", "pc", "neutral_cpc", "gated_cpc")


def forbidden(*args, **kwargs):
    raise AssertionError("invalid option must be refused before creating resources")


def protocol():
    return dict(
        protocol_id="fake_v1",
        metric_direction="loss lower; no live success",
        planned_work=dict(score_forwards=94, score_rows=846, maximum_source_environment_steps=0),
        evaluation="no live/final evaluation",
        limitations="fake fixture",
    )


@pytest.mark.parametrize("flag", [None, 0, 1, 0.0, 1.0, "true", [], {}])
def test_should_refuse_non_bool_extended_flag_before_budget_or_protocol_changes(monkeypatch, flag):
    monkeypatch.setattr(live, "ToyBudgetSession", forbidden)
    value = protocol()
    original = deepcopy(value)
    with pytest.raises(ValueError):
        live._live_budget(wide_neural=True, extended_wide_probe_budget=flag)
    with pytest.raises(ValueError):
        live.describe(value, wide_neural=True, extended_wide_probe_budget=flag)
    with pytest.raises(ValueError):
        live.ProbeRollouts(0, wide_method="bp", extended_wide_probe_budget=flag)
    assert value == original


@pytest.mark.parametrize("wide", [False, None, 0, 1, "true"])
def test_should_refuse_extended_budget_without_exact_wide_recipe_before_ports(monkeypatch, wide):
    monkeypatch.setattr(live, "ToyBudgetSession", forbidden)
    value = protocol()
    original = deepcopy(value)
    with pytest.raises(ValueError):
        live._live_budget(wide_neural=wide, extended_wide_probe_budget=True)
    with pytest.raises(ValueError):
        live.describe(value, wide_neural=wide, extended_wide_probe_budget=True)
    assert value == original


@pytest.mark.parametrize("method", [None, True, 1, "BP", "recurrent_bc", "pc_relax64"])
def test_should_refuse_extended_observer_without_declared_original_method(method):
    with pytest.raises(ValueError):
        live.ProbeRollouts(128, wide_method=method, extended_wide_probe_budget=True)


def test_should_preserve_default_descriptors_and_identify_only_selected_limits():
    implicit, explicit, extended = protocol(), protocol(), protocol()
    live.describe(implicit, wide_neural=True)
    live.describe(explicit, wide_neural=True, extended_wide_probe_budget=False)
    live.describe(extended, wide_neural=True, extended_wide_probe_budget=True)
    assert implicit == explicit
    assert implicit["probe_rollout"]["budget"] == (
        "separate40s zero training/replay updates per H32 endpoint inside original180s/RSS cap; count reader retains20s"
    )
    assert "endpoint_wall_seconds" not in implicit["probe_rollout"]
    assert "parent_wall_seconds" not in implicit["probe_rollout"]
    changed = extended["probe_rollout"]
    assert changed["endpoint_wall_seconds"] == 60.0
    assert changed["parent_wall_seconds"] == 240.0
    assert "60s" in changed["budget"] and "240s" in changed["budget"]
    changed.pop("endpoint_wall_seconds")
    changed.pop("parent_wall_seconds")
    changed["budget"] = implicit["probe_rollout"]["budget"]
    assert extended == implicit


@pytest.mark.parametrize(
    "wide,extended,seconds", [(False, False, 20.0), (True, False, 40.0), (True, True, 60.0)]
)
def test_should_bear_fixed_new_budget_limits_without_training_or_replay_authority(
    monkeypatch, wide, extended, seconds
):
    now = [1.0]
    monkeypatch.setattr(live, "monotonic", lambda: now[0])
    budget, binding = live._live_budget(wide_neural=wide, extended_wide_probe_budget=extended)
    assert budget.budget.max_wall_seconds == seconds
    assert budget.budget.max_training_updates == budget.budget.max_replay_examples == 0
    original = budget.budget
    now[0] += seconds - 0.5
    budget.before_final()
    binding()
    now[0] += 1.0
    with pytest.raises(ToyExecutionStopped):
        budget.before_final()
    binding()
    assert budget.budget is original
    assert budget.started_at == 1.0
    assert budget.last_clock == now[0]
    assert budget.updates_completed == budget.replay_examples_completed == 0


@pytest.mark.parametrize("fault", ["limit", "owner", "start", "clock"])
def test_should_refuse_replaced_or_unobserved_extended_budget_provenance(monkeypatch, fault):
    monkeypatch.setattr(live, "monotonic", lambda: 1.0)
    budget, binding = live._live_budget(wide_neural=True, extended_wide_probe_budget=True)
    if fault == "limit":
        object.__setattr__(budget.budget, "max_wall_seconds", 61.0)
    elif fault == "owner":
        budget.budget = replace(budget.budget)
    elif fault == "start":
        budget.started_at -= 0.5
    else:
        budget.last_clock += 0.5
    with pytest.raises(ValueError):
        binding()


def paid_report():
    return dict(
        status="complete",
        work=dict(
            environment_resets_completed=4,
            live_prediction_calls_returned=4,
            environment_steps_returned=4,
            internal_projection_rows_returned=4,
            internal_projection_array_bytes_returned=9472,
            failure_work_may_be_unobserved=False,
        ),
    )


@pytest.fixture
def fake_endpoints(monkeypatch):
    """Only orchestration/guards are real; no native snapshot claim is made."""
    calls = []
    monkeypatch.setattr(live, "monotonic", lambda: 1.0)
    monkeypatch.setattr(
        live, "_copy_wide_neural_probe_snapshot", lambda value, **options: deepcopy(value)
    )

    def evaluate(learner, state, **options):
        calls.append(options)
        options["check_boundary"]()
        return paid_report()

    monkeypatch.setattr(live, "evaluate_wide_neural_probe_checkpoint", evaluate)
    return SimpleNamespace(snapshot_state=lambda: {"fake_weights": (1, 2)}), calls


@pytest.mark.parametrize("method", METHODS)
def test_should_create_distinct_sixty_second_budgets_for_two_ordered_endpoints(
    fake_endpoints, method
):
    learner, calls = fake_endpoints
    observer = live.ProbeRollouts(128, wide_method=method, extended_wide_probe_budget=True)
    observer.observe(learner, "after_probe", lambda: None)
    first = calls[0]["budget"]
    observer.observe(learner, "after_return", lambda: None)
    second = calls[1]["budget"]
    observer.check()
    assert first is not second and first.budget is not second.budget
    for budget in (first, second):
        assert budget.budget.max_wall_seconds == 60.0
        assert budget.budget.max_training_updates == budget.budget.max_replay_examples == 0
        assert budget.started_at == budget.last_clock == 1.0
    assert [record["status"] for record in observer.records] == ["complete", "complete"]
    assert all(
        record["charged_updates"] == record["replay_examples"] == 0 for record in observer.records
    )


@pytest.mark.parametrize("change", [False, 1, None])
def test_should_refuse_changed_or_non_bool_owner_option_before_next_budget(
    fake_endpoints, monkeypatch, change
):
    learner, calls = fake_endpoints
    observer = live.ProbeRollouts(0, wide_method="bp", extended_wide_probe_budget=True)
    observer.observe(learner, "after_probe", lambda: None)
    observer.extended_wide_probe_budget = change
    monkeypatch.setattr(live, "_live_budget", forbidden)
    with pytest.raises(ValueError, match="configuration changed"):
        observer.observe(learner, "after_return", lambda: None)
    assert len(calls) == 1


@pytest.mark.parametrize("fault", ["report", "value", "budget", "clock"])
def test_should_validate_prior_extended_paid_evidence_before_more_work(
    fake_endpoints, monkeypatch, fault
):
    learner, calls = fake_endpoints
    observer = live.ProbeRollouts(0, wide_method="bp", extended_wide_probe_budget=True)
    observer.observe(learner, "after_probe", lambda: None)
    original = deepcopy(observer.records)
    if fault == "report":
        observer.records[0]["report"] = deepcopy(observer.records[0]["report"])
    elif fault == "value":
        observer.records[0]["report"]["work"]["environment_steps_returned"] += 1
    elif fault == "budget":
        calls[0]["budget"].budget = replace(calls[0]["budget"].budget)
    else:
        calls[0]["budget"].last_clock += 0.5
    monkeypatch.setattr(live, "_live_budget", forbidden)
    with pytest.raises(ValueError):
        observer.observe(learner, "after_return", lambda: None)
    assert len(calls) == 1
    observer.restore_paid()
    assert observer.records == original
