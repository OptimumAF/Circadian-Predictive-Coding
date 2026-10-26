"""Setup allowances fund disposable probes without renewing checkpoint budgets."""

from copy import deepcopy
from dataclasses import replace
import pickle
from typing import Any

import numpy as np
import pytest

import src.adapters.learnability_pilot as pilot
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.learner_ports import TrainingDiagnostic


class FakeLearner:
    def __init__(self, trace, *, value=0.0, hook=None):
        self.trace = trace
        self.value = np.array([value], dtype=np.float64)
        self.hook = hook

    def snapshot_state(self):
        return self.value.copy()

    def fork(self):
        self.trace.append(("fork",))
        return FakeLearner(self.trace, value=float(self.value[0]), hook=self.hook)

    def restore_state(self, state):
        self.value = state.copy()

    def predict(self, features):
        self.trace.append(("predict", len(features)))
        return np.full((len(features), 1), 0.5 + float(self.value[0]) / 100)

    def train_batch(self, features, targets):
        self.trace.append(("train", len(features)))
        self.value += 1
        if self.hook is not None:
            self.hook()
        return TrainingDiagnostic("prepared_fake_update", float(self.value[0]))


def setup(monkeypatch, *, hook=None):
    now = [1.0]

    def clock():
        return now[0]

    monkeypatch.setattr(pilot, "monotonic", clock)
    monkeypatch.setattr(
        pilot,
        "_measure_features",
        lambda state, reference, row: row.update(feature_health=[], diagnostic_seconds=0.0),
    )
    budgets = tuple(
        ToyBudgetSession(ToyExecutionBudget(max_training_updates=16, max_wall_seconds=59.0), clock)
        for _ in range(4)
    )
    rows = pilot.empty_learnability_observations()
    template = FakeLearner([], hook=hook)
    return now, budgets, rows, template


def create(template, rows, budgets, *, deadline=60.0):
    return pilot.create_sixteen_probe_observer(template, rows, deadline, probe_budgets=budgets)


def test_should_spend_four_original_allowances_on_owned_ordered_checkpoint_probes(monkeypatch):
    now, budgets, rows, template = setup(monkeypatch)
    original_limits = tuple(budget.budget for budget in budgets)
    original_starts = tuple(budget.started_at for budget in budgets)
    before = pickle.dumps(template.snapshot_state())
    observer = create(template, rows, budgets)
    # A new session at checkpoint time would hide the elapsed setup interval.
    now[0] = 7.0
    monkeypatch.setattr(pilot, "ToyBudgetSession", lambda *args, **kwargs: pytest.fail("renewed"))
    for index, point in enumerate((0, 6, 12, 18)):
        state = np.array([float(point)])
        checkpoint_before = pickle.dumps(state)
        observer(point, state, None)
        assert pickle.dumps(state) == checkpoint_before
        assert budgets[index].updates_completed == 16
        assert rows[index]["status"] == "complete"
        assert rows[index]["probe"]["completed_training_examples"] == 128
        assert rows[index]["probe"]["attempted_updates"] == 16
        assert rows[index]["probe"]["score_forward_calls_completed"] == 6
        assert [p["updates"] for p in rows[index]["probe"]["checkpoints"]] == [0, 1, 2, 4, 8, 16]
    assert pickle.dumps(template.snapshot_state()) == before
    assert all(
        budget.budget is limits for budget, limits in zip(budgets, original_limits, strict=True)
    )
    assert tuple(budget.started_at for budget in budgets) == original_starts
    assert sum(entry[0] == "train" for entry in template.trace) == 64
    with pytest.raises(RuntimeError, match="ordered"):
        observer(18, np.array([18.0]), None)
    assert rows[-1]["status"] == "complete"
    assert sum(entry[0] == "train" for entry in template.trace) == 64


@pytest.mark.parametrize(
    "fault", ["list", "short", "aliased", "spent", "wrong_quota", "clock", "rss", "expired"]
)
def test_should_refuse_invalid_prepared_allowances_before_reserved_data(monkeypatch, fault):
    now, budgets, rows, template = setup(monkeypatch)
    invalid: Any = budgets
    if fault == "list":
        invalid = list(budgets)
    elif fault == "short":
        invalid = budgets[:-1]
    elif fault == "aliased":
        invalid = (budgets[0],) * 4
    elif fault == "spent":
        budgets[1].record_update()
    elif fault == "wrong_quota":
        budgets[1].budget = replace(budgets[1].budget, max_training_updates=15)
    elif fault == "clock":
        budgets[1].clock = lambda: now[0]
    elif fault == "rss":
        budgets[1].process_rss_sampler = object()
    else:
        now[0] = 60.0
    monkeypatch.setattr(
        pilot, "build_reserved_data", lambda **kwargs: pytest.fail("data generated")
    )
    with pytest.raises(ValueError):
        create(template, rows, invalid)
    assert template.trace == []


@pytest.mark.parametrize("fault", ["limits", "start", "clock", "future_charge", "prior_reset"])
def test_should_refuse_changed_originals_before_next_probe_native_work(monkeypatch, fault):
    _, budgets, rows, template = setup(monkeypatch)
    observer = create(template, rows, budgets)
    observer(0, np.array([0.0]), None)
    calls = deepcopy(template.trace)
    if fault == "limits":
        budgets[1].budget = replace(budgets[1].budget)
    elif fault == "start":
        budgets[1].started_at += 1
    elif fault == "clock":
        budgets[1].clock = lambda: 1.0
    elif fault == "future_charge":
        budgets[2].record_update()
    else:
        budgets[0].updates_completed = 0
    with pytest.raises(RuntimeError, match="original budget"):
        observer(6, np.array([6.0]), None)
    assert template.trace == calls
    assert rows[1]["status"] == "failed" and rows[1]["probe"] is None


def test_should_preserve_paid_update_when_original_budget_replaced_by_native_callback(monkeypatch):
    _, budgets, rows, template = setup(monkeypatch)

    def replace_budget():
        budgets[0].budget = replace(budgets[0].budget)

    template.hook = replace_budget
    observer = create(template, rows, budgets)
    with pytest.raises(RuntimeError, match="fresh probe incomplete"):
        observer(0, np.array([0.0]), None)
    paid = rows[0]["probe"]
    assert rows[0]["status"] == "failed" and paid["status"] == "failed"
    assert paid["attempted_updates"] == paid["completed_updates"] == 1
    assert paid["completed_training_examples"] == 8
    assert budgets[0].updates_completed == 1
    assert paid["score_forward_calls_completed"] == 1
    assert "original budget" in paid["reason"]
    assert all(row["status"] == "not_run" for row in rows[1:])


def test_should_enforce_original_absolute_deadline_after_paid_update(monkeypatch):
    now, budgets, rows, template = setup(monkeypatch)
    # The session's own deadline may be later due to constructor clock overhead.
    observer = create(template, rows, budgets, deadline=10.0)
    template.hook = lambda: now.__setitem__(0, 10.0)
    with pytest.raises(RuntimeError, match="fresh probe incomplete"):
        observer(0, np.array([0.0]), None)
    paid = rows[0]["probe"]
    assert paid["completed_updates"] == budgets[0].updates_completed == 1
    assert paid["score_forward_calls_completed"] == 1
    assert "global deadline exhausted" in paid["reason"]


def test_should_refuse_out_of_order_checkpoint_before_paid_diagnostics_or_probe(monkeypatch):
    _, budgets, rows, template = setup(monkeypatch)
    observer = create(template, rows, budgets)
    monkeypatch.setattr(pilot, "_measure_features", lambda *args: pytest.fail("diagnostics paid"))
    with pytest.raises(RuntimeError, match="ordered"):
        observer(6, np.array([6.0]), None)
    assert template.trace == []
    assert all(budget.updates_completed == 0 for budget in budgets)
