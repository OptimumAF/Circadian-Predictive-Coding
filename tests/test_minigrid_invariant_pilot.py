"""Recipe integration uses native TRAIN counts and delegated fixture readouts."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from scripts.run_minigrid_invariant_pilot import load_inputs
from src.adapters import minigrid_invariant_pilot as pilot
from src.adapters import minigrid_learning_pilot as shared
from src.app import toy_execution_budget as budgets
from test_torch_feature_rehearsal_pilot import Sampler, forbidden


@pytest.fixture(scope="module")
def admitted():
    return load_inputs(
        Path(__file__).resolve().parents[1]
        / "artifacts/runs/minigrid-action-memory-live-20261017/native.json"
    )


@pytest.fixture
def case(monkeypatch, admitted):
    states, sources = deepcopy(admitted)
    monkeypatch.setattr(pilot, "ProcessRssSampler", Sampler)
    monkeypatch.setattr(budgets, "ProcessRssSampler", Sampler)
    monkeypatch.setattr(shared, "require_torch", forbidden)
    monkeypatch.setattr(shared, "_same_state", forbidden)
    calls: list[dict[str, Any]] = []

    def evaluate(source, state, *, updates, budget, check_boundary):
        check_boundary()
        assert budget.budget.max_training_updates == budget.updates_completed == 0
        assert state.completed_updates == updates
        value = dict(
            status="complete",
            updates=updates,
            episodes=[
                dict(
                    context=context,
                    success=bool(updates),
                    **{"return": float(bool(updates))},
                    environment=dict(environment_steps=2),
                )
                for context in ("A", "A", "E-blue", "E-red")
            ],
            work=dict(fixture_only=True),
        )
        calls.append(value)
        return value

    monkeypatch.setattr(pilot, "evaluate_representation_checkpoint", evaluate)
    return states, sources, calls, evaluate


def test_should_run_native_A32_before_readouts_and_leave_originals_unchanged(case):
    states, sources, calls, _ = case
    before = deepcopy((states, sources))
    report = pilot.run_pilot(states, sources)
    assert report["status"] == "complete", report["reason"]
    training = report["training"]
    assert training["charged_updates"] == 32
    assert training["native_training_examples"] == 288
    assert len(training["partial_stream"]["applied_updates"]) == 32
    assert training["source_unchanged"] is training["actor_unchanged"] is True
    assert training["stream_result"]["score_forward_calls"] == 0
    assert [(value["representation"], value["updates"]) for value in report["readouts"]] == [
        ("raw", 0),
        ("raw", 32),
        ("invariant", 0),
        ("invariant", 32),
    ]
    assert len(calls) == 4
    assert report["readout_training_updates"] == 0
    assert report["training_projection_accounting"]["projected_array_bytes_returned"] == 288 * 2368
    assert (states, sources) == before


def test_should_refuse_wrong_original_count_binding_before_any_training(monkeypatch, case):
    states, sources, calls, _ = case
    entry = states[1].entries[0]
    counts = list(entry.counts)
    action = next(index for index, count in enumerate(counts) if count)
    counts[action] -= 1
    counts[(action + 1) % 7] += 1
    bad = replace(states[1], entries=(replace(entry, counts=tuple(counts)), *states[1].entries[1:]))
    monkeypatch.setattr(shared, "_train_arm", forbidden)
    with pytest.raises(ValueError, match="counts differ"):
        pilot.run_pilot((states[0], bad), sources)
    assert not calls


@pytest.mark.parametrize("fault", ["lost", "prior", "input", "training"])
def test_should_stop_on_callback_fault_and_preserve_paid_prefix(monkeypatch, case, fault):
    states, sources, calls, evaluate = case
    training = None
    original_train = shared._train_arm

    def train(*args, **kwargs):
        nonlocal training
        arm = original_train(*args, **kwargs)
        training = arm["row"]
        return arm

    def corrupt(source, state, **kwargs):
        if calls:
            if fault == "lost":
                raise RuntimeError("lost delegated report")
            if fault == "prior":
                calls[0]["episodes"][0]["success"] = True
            if fault == "input":
                object.__setattr__(states[1], "completed_updates", 64)
            if fault == "training":
                assert training is not None
                training["charged_updates"] = 0
            kwargs["check_boundary"]()
        return evaluate(source, state, **kwargs)

    monkeypatch.setattr(shared, "_train_arm", train)
    monkeypatch.setattr(pilot, "evaluate_representation_checkpoint", corrupt)
    report = pilot.run_pilot(states, sources)
    assert report["status"] == "incomplete"
    assert len(calls) == 1
    assert report["training"]["charged_updates"] == 32
    assert report["readouts"][0]["status"] == "complete"
    assert report["readouts"][0]["episodes"][0]["success"] is False
    assert report["readouts"][1]["delegated_report_unavailable"] is True
    assert report["readouts"][1]["work"]["environment_steps_attempted"] is None
    assert all(slot["status"] == "not_run" for slot in report["readouts"][2:])
    assert report["contrasts"] == {}


@pytest.mark.parametrize("port", ["guard", "clock", "exit", "snapshot"])
def test_should_refuse_late_parent_callbacks_before_publication(monkeypatch, case, port):
    states, sources, calls, _ = case
    original_train = shared._train_arm
    triggered = False

    def corrupt():
        nonlocal triggered
        if calls and not triggered:
            triggered = True
            calls[0]["episodes"][0]["success"] = True

    if port == "guard":

        def train(*args, **options):
            arm = original_train(*args, **options)
            original = arm["guard"]

            def guard():
                original()
                corrupt()

            arm["guard"] = guard
            return arm

        monkeypatch.setattr(shared, "_train_arm", train)
    elif port == "clock":
        original = budgets.ToyBudgetSession.before_final

        def before_final(owner):
            original(owner)
            if owner.budget.max_training_updates == 0:
                corrupt()

        monkeypatch.setattr(budgets.ToyBudgetSession, "before_final", before_final)
    else:

        class ClosingSampler(Sampler):
            closed = False

            def __exit__(self, *args):
                self.closed = True
                if port == "exit":
                    corrupt()
                return False

            def snapshot(self):
                value = super().snapshot()
                if self.closed and port == "snapshot":
                    corrupt()
                return value

        monkeypatch.setattr(pilot, "ProcessRssSampler", ClosingSampler)
        monkeypatch.setattr(budgets, "ProcessRssSampler", ClosingSampler)
    report = pilot.run_pilot(states, sources)
    assert triggered
    assert report["status"] == "incomplete"
    assert report["training"]["charged_updates"] == 32
    assert report["readouts"][0]["episodes"][0]["success"] is False
    assert report["contrasts"] == {}
