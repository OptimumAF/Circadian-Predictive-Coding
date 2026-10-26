"""Opt-in serving composition with scripted ports; no actual learner or teacher."""

from copy import deepcopy
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scripts import context_head_serving as demo
from scripts import run_bp_ticket_serving as cli
from src.app import arrived_shift_observer as observer_module
from src.app.coupled_attribution import AttributionBatch
from src.core.learner_ports import TrainingDiagnostic


@dataclass(frozen=True)
class State:
    policy: str = "scripted"
    completed_updates: int = 12
    training_examples: int = 72


@dataclass(frozen=True)
class Serving:
    model_state: State
    generation: int = 0


@dataclass(frozen=True)
class Guard:
    accepted: bool


class Rejected(Exception):
    def __init__(self, report):
        self.report = report


class Learner:
    def __init__(self, policy):
        self.policy = policy
        self.state = State(policy)

    def restore_state(self, state):
        self.state = deepcopy(state)

    def snapshot_state(self):
        return deepcopy(self.state)

    def train_batch(self, features, targets):
        self.state = replace(
            self.state,
            completed_updates=self.state.completed_updates + 1,
            training_examples=self.state.training_examples + len(features),
        )
        return TrainingDiagnostic("scripted_only", float(len(features)))


class Budget:
    def __init__(self, limits, clock):
        self.budget, self.clock = limits, clock
        self.started_at = self.last_clock = 0.0
        self.updates_completed = 0

    def attach_memory(self, sampler):
        pass

    def before_final(self):
        pass


class Actor:
    def __init__(self, source, **kwargs):
        self.initial = self.state = source.snapshot_state()
        self.generation = 0

    def serving_snapshot(self):
        return Serving(self.state, self.generation)


class Controller:
    def __init__(self, actor, **kwargs):
        self.actor = actor

    def prepare(self, runtime, *args, now, **kwargs):
        if now == 1:
            raise Rejected(Guard(False))
        return SimpleNamespace(report=Guard(True))

    def discard(self, ticket):
        pass

    def commit(self, runtime, ticket):
        self.actor.state = runtime.candidate_snapshot()
        self.actor.generation += 1
        return Guard(True)

    def rollback(self, receipt):
        self.actor.state = self.actor.initial
        self.actor.generation += 1
        return self.actor.generation


class Stream:
    def __init__(self, source, training, heldout, *, budget, actor, **kwargs):
        self.candidate = source.fork()
        self.training, self.budget = training, budget
        self.observer = kwargs.get("observe_checkpoint")
        self.updates = 0
        self._runtime = SimpleNamespace(
            applied_updates=[], candidate_snapshot=self.candidate_snapshot
        )
        if self.observer is not None:
            self.observer(0, self.candidate_snapshot(), None)

    def candidate_snapshot(self):
        return self.candidate.snapshot_state()

    def run_until(self, point):
        while self.updates < point:
            batch = self.training[self.updates]
            self.candidate.train_batch(batch.features, batch.targets)
            self.updates += 1
            self.budget.updates_completed += 1
            if self.observer is not None:
                self.observer(self.updates, self.candidate_snapshot(), batch)

    def finish(self):
        return {"scripted_updates": self.updates}


@pytest.fixture
def ports(monkeypatch):
    monkeypatch.setattr(demo, "ContextPerceptronLearner", Learner)
    monkeypatch.setattr(demo, "ToyBudgetSession", Budget)
    monkeypatch.setattr(demo, "PromotableActor", Actor)
    monkeypatch.setattr(demo, "AttributionStream", Stream)
    monkeypatch.setattr(demo, "ServingPromotionController", Controller)
    monkeypatch.setattr(demo, "PromotionRejected", Rejected)
    monkeypatch.setattr(demo, "categorical_guard", lambda *a: None)
    monkeypatch.setattr(
        demo, "ContextServingPredictor", lambda actor, spec: SimpleNamespace(actor=actor, work={})
    )
    monkeypatch.setattr(observer_module, "monotonic", lambda: 0.0)

    def episodes(predictor, phase, require, rows):
        require()
        rows.append(dict(phase=phase, complete=True, generation=predictor.actor.generation))

    monkeypatch.setattr(demo, "_episodes", episodes)
    calls = []

    def predict(state, features):
        calls.append((state.completed_updates, len(features)))
        return np.full((len(features), 6), 1 / 6)

    monkeypatch.setattr(demo, "predict_context_perceptron", predict)
    return calls


def arm(mode="guarded", **kwargs):
    packets = tuple(
        AttributionBatch(
            "train",
            "C",
            tuple(f"row-{i}-{n}" for n in range(rows)),
            np.zeros((rows, 9)),
            np.zeros((rows, 1)),
            num_classes=6,
        )
        for i, rows in enumerate((3, 6))
    )
    row: dict[str, Any] = dict(episodes=[], guards=[], candidate_states=[], publications=[])
    demo.run_arm(mode, State(), None, (packets, (None, None)), None, lambda: None, row, **kwargs)
    return row


def test_should_observe_previous_native_states_and_preserve_default_complete_arm(ports):
    baseline = arm()
    assert ports == [] and "shift_observations" not in baseline
    observed = arm(shift_diagnostics=True)
    assert ports == [(12, 3), (13, 6)]
    diagnostics = observed.pop("shift_observations")
    assert observed == baseline
    assert diagnostics["native_update_origin"] == 12
    report = diagnostics["report"]
    assert report["prediction_schema"]["prediction_rows_completed"] == 9
    assert report["prediction_calls_completed"] == 2
    assert [r["prediction_state_updates"] for r in report["rows"]] == [0, 1]
    assert all(
        r["decision"]["phase"] == "reference" and not any(r["decision"]["alarms"])
        for r in report["rows"]
    )
    assert observed["source_unchanged"] and observed["rollback_state_equal_source"]
    assert observed["budget"]["updates"] == 2
    assert [g["report"]["accepted"] for g in observed["guards"]] == [False, True]


def test_should_not_construct_or_predict_diagnostics_in_frozen_arm(ports, monkeypatch):
    monkeypatch.setattr(
        demo, "ArrivedShiftObserver", lambda *a, **k: pytest.fail("frozen observer")
    )
    row = arm("frozen", shift_diagnostics=True)
    assert row["complete"] and "shift_observations" not in row and ports == []


def test_should_preserve_paid_failed_observer_and_stop_before_second_publication(
    ports, monkeypatch
):
    rows = []

    def fail(state, features):
        rows.append(len(features))
        if len(rows) == 2:
            raise RuntimeError("diagnostic failed")
        return np.full((len(features), 6), 1 / 6)

    monkeypatch.setattr(demo, "predict_context_perceptron", fail)
    row: dict[str, Any] = dict(episodes=[], guards=[], candidate_states=[], publications=[])
    packets = tuple(
        AttributionBatch(
            "train", "C", (f"r{i}",), np.zeros((1, 9)), np.zeros((1, 1)), num_classes=6
        )
        for i in range(2)
    )
    with pytest.raises(RuntimeError, match="diagnostic failed"):
        demo.run_arm(
            "guarded",
            State(),
            None,
            (packets, (None, None)),
            None,
            lambda: None,
            row,
            shift_diagnostics=True,
        )
    report = row["shift_observations"]["report"]
    assert report["stopped"] and [r["status"] for r in report["rows"]] == ["complete", "failed"]
    assert report["prediction_calls_attempted"] == 2 and report["prediction_calls_completed"] == 1
    assert row["training_work"]["returned_updates"] == row["budget"]["updates"] == 2
    assert len(row["guards"]) == 1 and "receipt" not in row and row["source_unchanged"]


@pytest.mark.parametrize("value", [1, 0, None, "yes"])
def test_should_refuse_nonboolean_opt_in_before_resources_or_models(monkeypatch, value):
    monkeypatch.setattr(demo, "ProcessRssSampler", lambda *a, **k: pytest.fail("sampler"))
    monkeypatch.setattr(demo, "ToyBudgetSession", lambda *a, **k: pytest.fail("budget"))
    with pytest.raises(ValueError, match="exact bool"):
        demo.run_case("unused", shift_diagnostics=value)
    with pytest.raises(ValueError, match="exact bool"):
        demo.run_arm("guarded", None, None, None, None, None, {}, shift_diagnostics=value)


@pytest.mark.parametrize("enabled", [False, True])
def test_should_route_only_explicit_flag_and_preserve_cli_default_kwargs(
    tmp_path, monkeypatch, enabled
):
    calls = []

    def run(path, **kwargs):
        calls.append(kwargs)
        return dict(complete=True)

    monkeypatch.setattr(demo, "run_case", run)
    args = ["--source", "unused", "--context-head-serving", "--output", str(tmp_path / "out.json")]
    if enabled:
        args.append("--context-shift-diagnostics")
    assert cli.main(args) == 0 and calls == ([dict(shift_diagnostics=True)] if enabled else [{}])


@pytest.mark.parametrize("mode", [[], ["--context-head-serving-live"], ["--reference", "drift"]])
def test_should_refuse_wrong_cli_mode_before_source_lookup_or_output(tmp_path, mode):
    output = tmp_path / "missing.json"
    with pytest.raises(SystemExit) as error:
        cli.main(
            ["--source", "missing", "--output", str(output), "--context-shift-diagnostics", *mode]
        )
    assert error.value.code == 2 and not output.exists()
