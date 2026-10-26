"""Train-role live diagnostics with owned fake state and actual toy dynamics."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from src.adapters import key_door_pilot as shared
from src.adapters import train_policy_diagnostics as diagnostic
from src.adapters.torch_recurrent_learner import TorchRecurrentPolicy
from src.app.coupled_attribution import _same_value
from src.app.key_door_policy_evaluation import KeyDoorEvaluationFailed
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget


@dataclass
class State:
    policy: TorchRecurrentPolicy
    completed_updates: int
    training_examples: int
    parameters: dict[str, float]


def state(point):
    return State(TorchRecurrentPolicy(8, 8, 5, 17), point, point * 8, {"weight": 1.0})


class Learner:
    def __init__(self, learned=None, trace=None, hook=None):
        self.state = deepcopy(state(0) if learned is None else learned)
        self.trace = [] if trace is None else trace
        self.hook = hook

    def fire(self, event):
        self.trace.append(event)
        if self.hook is not None:
            self.hook(event, self)

    def snapshot_state(self):
        self.fire("snapshot")
        return deepcopy(self.state)

    def fork(self):
        self.fire("fork")
        return Learner(self.state, self.trace, self.hook)

    def restore_state(self, learned):
        self.state = deepcopy(learned)
        self.fire("restore")

    def start_episode(self):
        self.fire("session")
        return Session(self.state, self.trace, self.hook)


class Session(Learner):
    completed_steps = 0

    def predict(self, features):
        self.fire("predict")
        self.completed_steps += 1
        return np.array([[0.2] * 5], dtype=np.float64)


def reference(probabilities=None):
    return dict(
        source_role="train",
        probabilities=[0.5, 0.125, 0, 0.125, 0.25] if probabilities is None else probabilities,
        class_counts=[8, 2, 0, 2, 4],
    )


def budget():
    result = ToyBudgetSession(ToyExecutionBudget(max_training_updates=64), lambda: 0.0)
    result.updates_completed = 64
    return result


@pytest.fixture(autouse=True)
def pure_state_comparison(monkeypatch):
    monkeypatch.setattr(diagnostic, "_same_state", _same_value)
    monkeypatch.setattr(shared, "_same_state", _same_value)


def execute(*, prior=None, hook=None, guard=None):
    report, source, states = (
        diagnostic.new_diagnostic(),
        Learner(hook=hook),
        [(0, state(0)), (64, state(64))],
    )
    saved_prior = reference() if prior is None else prior
    original = deepcopy((source.state, states, saved_prior))
    session = budget()
    diagnostic.evaluate_train_policy(
        report, source, states, saved_prior, session, check_boundary=guard or (lambda: None)
    )
    assert _same_value((source.state, states, saved_prior), original)
    assert session.updates_completed == 64
    return report, source


def test_should_allocate_detached_six_train_slots_without_resources():
    first, second = diagnostic.new_diagnostic(), diagnostic.new_diagnostic()
    assert first["source_role"] == "train" and first["classification"] == "train_resubstitution"
    assert [(s["checkpoint_updates"], s["cue"]) for s in first["episodes"]] == [
        (0, 1),
        (0, -1),
        (64, 1),
        (64, -1),
    ]
    assert len(first["reference"]["episodes"]) == 2
    first["episodes"][0]["status"] = "changed"
    assert second["episodes"][0]["status"] == "not_run"


def test_should_run_four_owned_legal_learner_episodes_and_two_fixed_prior_episodes():
    report, source = execute(prior=reference([0.2] * 5))
    assert report["status"] == "complete"
    assert source.trace.count("fork") == 2 and source.trace.count("session") == 4
    assert source.trace.count("predict") == 48
    assert report["work"]["learner_sessions_started"] == 4
    assert report["work"]["learner_callback_returns"] == 48
    assert report["work"]["prior_callback_returns"] == 24
    assert report["reference"]["outcomes"] == {"success_rate": 0.0, "mean_return": 0.0}
    assert report["learner_gains"] == {"success_rate": 0.0, "mean_return": 0.0}
    assert report["comparison"] == [
        dict(updates=n, success_rate=0.0, mean_return=0.0) for n in (0, 64)
    ]
    for slot in (*report["episodes"], *report["reference"]["episodes"]):
        assert slot["source_role"] == "train" and slot["action_conditioning"] == "legal_actions_v1"
        assert slot["state_unchanged"] is True
        assert all(t["raw_probabilities"] == (0.2,) * 5 for t in slot["transitions"])


def test_should_preserve_saved_zero_support_prior_refusals_without_imputed_comparison():
    report, _ = execute()
    assert report["status"] == "complete" and len(report["outcomes"]) == 2
    assert report["comparison"] is None and "zero_legal_mass" in report["comparison_reason"]
    assert report["reference"]["outcomes"] is None
    for slot in report["reference"]["episodes"]:
        assert slot["status"] == "refused" and slot["refusal"] == "zero_legal_mass"
        assert slot["success"] is None and slot["checkpoint_origin"] == "fixed_train_label_prior"
        assert slot["transitions"][-1]["action"] is None
        assert slot["transitions"][-1]["probabilities"] is None
        assert slot["prediction_calls_returned"] == slot["steps_executed"] + 1
        assert slot["return"] == sum(t["reward"] for t in slot["transitions"])


@pytest.mark.parametrize(
    "fault", ["point", "bool", "rows", "updates", "policy", "shape", "container"]
)
def test_should_refuse_wrong_checkpoints_before_source_work(fault):
    states: Any = [(0, state(0)), (64, state(64))]
    if fault == "point":
        states[1] = (63, state(64))
    elif fault == "bool":
        states[0] = (False, state(0))
    elif fault == "rows":
        states[1][1].training_examples = 511
    elif fault == "updates":
        states[1][1].completed_updates = True
    elif fault == "policy":
        states[1][1].policy = TorchRecurrentPolicy(8, 9, 5, 17)
    elif fault == "shape":
        states[1] = [64, state(64)]
    else:
        states = tuple(states)
    source = Learner()
    with pytest.raises(ValueError, match="checkpoint"):
        diagnostic.evaluate_train_policy(
            diagnostic.new_diagnostic(),
            source,
            states,
            reference(),
            budget(),
            check_boundary=lambda: None,
        )
    assert source.trace == []


@pytest.mark.parametrize(
    "fault", ["role", "nan", "negative", "sum", "bool", "budget", "slots", "callback"]
)
def test_should_refuse_invalid_prior_budget_or_report_before_native_work(fault):
    report, prior, session, source = diagnostic.new_diagnostic(), reference(), budget(), Learner()
    guard: Any = lambda: None
    if fault == "role":
        prior["source_role"] = "final_test"
    elif fault == "nan":
        prior["probabilities"][0] = float("nan")
    elif fault == "negative":
        prior["probabilities"][0] = -0.5
    elif fault == "sum":
        prior["probabilities"][0] = 0.4
    elif fault == "bool":
        prior["probabilities"][0] = True
    elif fault == "budget":
        session.budget = ToyExecutionBudget(max_training_updates=65)
    elif fault == "slots":
        report["episodes"].pop()
    else:
        guard = None
    with pytest.raises(ValueError):
        diagnostic.evaluate_train_policy(
            report, source, [(0, state(0)), (64, state(64))], prior, session, check_boundary=guard
        )
    assert source.trace == []


@pytest.mark.parametrize(
    "target", ["state", "prior", "preceding", "source", "unreached_cue", "replacement"]
)
def test_should_stop_before_more_work_when_prior_or_preceding_evidence_changes(target):
    report, states, prior, source = (
        diagnostic.new_diagnostic(),
        [(0, state(0)), (64, state(64))],
        reference(),
        Learner(),
    )
    changed = False

    def corrupt(event, actor):
        nonlocal changed
        if event == "predict" and not changed:
            changed = True
            if target == "state":
                states[1][1].parameters["weight"] = 2.0
            elif target == "prior":
                prior["probabilities"][0] = 0.4
            elif target == "source":
                source.state.parameters["weight"] = 2.0
            elif target == "unreached_cue":
                report["episodes"][-1]["cue"] = 1
            elif target == "replacement":
                report["episodes"][-1] = deepcopy(report["episodes"][-1])

    source.hook = corrupt

    def guard():
        if target == "preceding" and report["episodes"][0]["state_unchanged"] is True:
            report["episodes"][0]["return"] = 7.0

    expected = ValueError if target == "preceding" else KeyDoorEvaluationFailed
    with pytest.raises(expected, match="changed") as caught:
        diagnostic.evaluate_train_policy(
            report, source, states, prior, budget(), check_boundary=guard
        )
    if expected is KeyDoorEvaluationFailed:
        assert type(caught.value.__cause__) is ValueError
    assert report["status"] == "failed"
    assert all(s["status"] == "not_run" for s in report["reference"]["episodes"])
    assert source.trace.count("session") < 4


def test_should_not_classify_unrelated_prior_predictor_failure_as_zero_mass(monkeypatch):
    original = diagnostic.run_key_door_episode

    def fail(predict, **options):
        return original(
            lambda _: (_ for _ in ()).throw(RuntimeError("injected prior fault")), **options
        )

    monkeypatch.setattr(diagnostic, "run_key_door_episode", fail)
    report = diagnostic.new_diagnostic()
    with pytest.raises(KeyDoorEvaluationFailed, match="injected prior fault"):
        diagnostic.evaluate_train_policy(
            report,
            Learner(),
            [(0, state(0)), (64, state(64))],
            reference(),
            budget(),
            check_boundary=lambda: None,
        )
    assert report["status"] == "failed" and report["reference"]["episodes"][0]["status"] == "failed"
    assert report["reference"]["episodes"][1]["status"] == "not_run"
    assert report["outcomes"] is not None and report["comparison"] is None


@pytest.mark.parametrize(
    "fault",
    ["raw", "action", "conditional", "applied", "count", "reward", "observation", "prefix_support"],
)
def test_should_stop_on_contradictory_prior_refusal_evidence(monkeypatch, fault):
    original = diagnostic.run_key_door_episode

    def corrupt(predict, **options):
        try:
            return original(predict, **options)
        except KeyDoorEvaluationFailed as error:
            record = error.report
            pending = record["transitions"][-1]
            if fault == "raw":
                pending["raw_probabilities"] = (float("nan"), 0, 0, 0, 1)
            elif fault == "action":
                pending["action"] = 2
            elif fault == "conditional":
                pending["probabilities"] = (0.2,) * 5
            elif fault == "applied":
                pending["applied"] = True
            elif fault == "count":
                record["prediction_calls_returned"] -= 1
            elif fault == "reward":
                pending["reward"] = 1
            elif fault == "prefix_support":
                # Adding zero-mass action2 preserves the conditional vector,
                # but the actual start observation declares only action0.
                record["transitions"][0]["legal_actions"] = (0, 2)
            else:
                pending["next_observation"] = (0,) * 8
            raise

    monkeypatch.setattr(diagnostic, "run_key_door_episode", corrupt)
    report = diagnostic.new_diagnostic()
    with pytest.raises(KeyDoorEvaluationFailed):
        diagnostic.evaluate_train_policy(
            report,
            Learner(),
            [(0, state(0)), (64, state(64))],
            reference(),
            budget(),
            check_boundary=lambda: None,
        )
    assert report["reference"]["episodes"][0]["status"] == "failed"
    assert report["reference"]["episodes"][1]["status"] == "not_run"
    assert report["comparison"] is None


def test_should_refuse_forged_zero_support_that_disagrees_with_actual_pending_observation():
    raw = (0.5, 0.125, 0, 0.125, 0.25)
    observation = (0.0, -1.0, -1.0, -1.0, 0.0, 0.0, 1.0, 0.0)
    record = dict(
        status="failed",
        terminal=False,
        success=False,
        prediction_calls_attempted=1,
        prediction_calls_returned=1,
        steps_executed=0,
        **{"return": 0.0},
        transitions=[
            dict(
                raw_probabilities=raw,
                probabilities=None,
                action=None,
                applied=False,
                observation=observation,
                next_observation=observation,
                reward=0.0,
                terminal=False,
                legal_actions=(2,),
            )
        ],
    )
    assert diagnostic._zero_legal_mass(record, raw) is False
