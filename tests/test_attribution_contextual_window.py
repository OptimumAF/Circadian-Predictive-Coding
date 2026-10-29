"""Real four-trial owner and finite stream, with only literal learner work."""

from copy import deepcopy
from dataclasses import replace
import json
import pickle
import random
from typing import Any

import numpy as np
import pytest

from src.app.coupled_attribution import AttributionBatch, AttributionStream
from src.app.counterfactual_controller import CounterfactualController, CounterfactualReturn
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.actor_ports import VersionedState
from src.core.controller_objective import ControllerMeasures, ControllerObjective
from src.core.experience import LogicalClock
from src.core.learner_ports import TrainingDiagnostic
from src.core.reference_controller import (
    ArrivedTrainingFeedback,
    ReferenceObservation,
    ReferencePolicy,
)

WORK = dict(owners=0, snapshots=0, updates=0, rows=0, predicts=0, predict_rows=0)


class LiteralLearner:
    def __init__(self, state=None, hooks=None, calls=None):
        WORK["owners"] += 1
        self.state: dict[str, Any] = (
            dict(weights=[0], optimizer=[0], rng=("literal", 0))
            if state is None
            else deepcopy(state)
        )
        self.hooks = {} if hooks is None else hooks
        self.calls = [] if calls is None else calls

    def fork(self):
        return LiteralLearner(self.state, self.hooks, self.calls)

    def snapshot_state(self):
        WORK["snapshots"] += 1
        return deepcopy(self.state)

    def restore_state(self, state):
        self.state = deepcopy(state)

    def train_batch(self, features, targets):
        WORK["updates"] += 1
        WORK["rows"] += len(features)
        self.calls.append(("train", float(features[0, 0])))
        self.state["weights"][0] += 1
        self.state["optimizer"][0] += 2
        if "train" in self.hooks:
            self.hooks["train"]()
        return TrainingDiagnostic("literal_only", float(self.state["weights"][0]))

    def predict(self, features):
        WORK["predicts"] += 1
        WORK["predict_rows"] += len(features)
        self.calls.append(("predict", float(features[0, 0])))
        if "predict" in self.hooks:
            self.hooks["predict"]()
        return np.zeros((len(features), 1), dtype=np.float64)


def make(*, target=1.0, delay=1, credit=4, final=False, observe=None):
    source = LiteralLearner()
    now = [0.0]
    budget = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=credit, max_wall_seconds=60), lambda: now[0]
    )
    batches = tuple(
        AttributionBatch(
            "train", "A", (f"train-{i}",), np.array([[float(i)]]), np.array([[target]])
        )
        for i in range(4)
    )
    heldout = (
        (AttributionBatch("final_test", "A", ("final",), np.array([[8.0]]), np.array([[0.0]])),)
        if final
        else ()
    )
    stream = AttributionStream(
        source,
        batches,
        heldout,
        checkpoint_updates=(0, 1, 2, 3, 4),
        label_delay=delay,
        budget=budget,
        training_only=not final,
        observe_checkpoint=observe,
    )
    return stream, source, budget, now


def fit_for(stream, source, *, clean="wait", mistakes="wake1", freeze=True, digest=json.dumps):
    current = [VersionedState(stream._runtime._base_actor_version, source.snapshot_state())]
    hook: list[Any] = []

    def read():
        if hook:
            hook.pop(0)()
        return deepcopy(current[0])

    clock = LogicalClock()
    trainer = CounterfactualController(read, digest, ControllerObjective(), clock)
    for context, preferred in (("clean", clean), ("mistakes", mistakes)):
        for action in ("wait", "wake1"):
            obs = ReferenceObservation(
                clock.now(),
                1,
                1,
                1,
                current[0].actor_version,
                ArrivedTrainingFeedback(
                    "train",
                    "fit",
                    current[0].actor_version,
                    0,
                    clock.now(),
                    int(context == "mistakes"),
                    1,
                ),
            )

            def operation(request, state):
                state["optimizer"][0] = 999  # disposable copy, never application learner
                clock.advance_to(request.started_at + 1)
                return CounterfactualReturn(
                    request,
                    clock.now(),
                    clock.now(),
                    ControllerMeasures(0, int(action == preferred), 1, 1, 1, 1, 1.0, 1024, 0, True),
                )

            trainer.run(obs, operation)
    if freeze:
        trainer.freeze()
    return trainer, current, clock, hook


@pytest.mark.parametrize("delay", [0, 1, 2])
@pytest.mark.parametrize("action", ["wait", "wake1"])
def test_should_apply_actual_fitted_choices_with_original_causal_fifo_and_separate_fit_costs(
    delay, action
):
    stream, source, budget, _ = make(delay=delay)
    trainer, _, clock, _ = fit_for(stream, source, mistakes=action)
    original = deepcopy(source.snapshot_state()), stream._runtime.actor.snapshot_state()
    rng = pickle.dumps((random.getstate(), np.random.get_state()))
    result = stream.run_contextual_window(trainer)
    n = 4 if action == "wake1" else 0
    assert result["window_complete"] is True and result["training_complete"] is (n == 4)
    assert (
        result["completed_updates"] == budget.updates_completed == n
        and result["pending_unapplied"] == 4 - n
    )
    assert result["training_examples"] == n and result["score_forward_calls"] == 0
    assert [r["applied_at"] for r in result["applied_updates"]] == list(range(delay, delay + n))
    assert [d["observation"]["tick"] for d in result["decisions"]] == list(range(4 + delay))
    fit = result["controller_application"]["fit"]
    assert (
        fit["attempts"] == fit["completed"] == fit["discarded_trials"] == 4
        and len(fit["returns"]) == 4
    )
    assert sum(r["measures"]["execution_seconds"] for r in fit["returns"]) == 4.0
    assert all(r["request"]["role"] == "controller_train" for r in fit["returns"])
    assert (
        result["policy"]["fitted_at"] == clock.now() == 4
        and result["decisions"][0]["observation"]["tick"] == 0
    )
    assert (
        result["controller_application"]["binding_reads_attempted"]
        == result["controller_application"]["binding_reads_completed"]
        > 4
    )
    assert not result["controller_application"]["unreturned_binding_unknown"]
    assert (source.snapshot_state(), stream._runtime.actor.snapshot_state()) == original
    assert pickle.dumps((random.getstate(), np.random.get_state())) == rng
    assert trainer.snapshot()["attempts"] == 4  # no fitting or fifth trial during application


def test_should_keep_correct_oldest_queued_and_respect_hard_credit():
    stream, source, budget, _ = make(target=0.0)
    trainer, _, _, _ = fit_for(stream, source)
    result = stream.run_contextual_window(trainer)
    assert result["completed_updates"] == budget.updates_completed == 0
    assert [e["batch_index"] for e in result["ready_feedback"]["events"]] == [0] * 4
    assert source.calls == [("predict", 0.0)] * 4 and result["pending_unapplied"] == 4
    stream2, source2, budget2, _ = make(credit=1)
    trainer2, _, _, _ = fit_for(stream2, source2)
    result2 = stream2.run_contextual_window(trainer2)
    assert (
        result2["completed_updates"] == budget2.updates_completed == 1
        and result2["pending_unapplied"] == 3
    )


@pytest.mark.parametrize("field", ["weights", "optimizer", "rng", "version"])
def test_should_refuse_different_complete_source_even_with_constant_digest(field):
    stream, source, budget, _ = make()
    trainer, current, _, _ = fit_for(stream, source, digest=lambda state: "constant")
    if field == "version":
        current[0] = replace(current[0], actor_version="other")
    elif field == "rng":
        current[0].state[field] = ("changed", 1)
    else:
        current[0].state[field][0] += 1
    before = WORK["predicts"], WORK["updates"]
    with pytest.raises(ValueError):
        stream.run_contextual_window(trainer)
    assert (WORK["predicts"], WORK["updates"]) == before and budget.updates_completed == 0


@pytest.mark.parametrize(
    "fault", ["policy", "returns", "measure_replace", "measure_value", "bool_cost", "objective"]
)
def test_should_refuse_changed_original_fit_evidence_before_any_application_work(fault):
    stream, source, _, _ = make()
    trainer, _, _, _ = fit_for(stream, source)
    if fault == "policy":
        trainer._fitted = deepcopy(trainer._fitted)
    elif fault == "returns":
        trainer._returns = deepcopy(trainer._returns)
    elif fault == "measure_replace":
        object.__setattr__(trainer._returns[0], "measures", deepcopy(trainer._returns[0].measures))
    elif fault == "measure_value":
        object.__setattr__(trainer._returns[0].measures, "execution_seconds", 2.0)
    elif fault == "bool_cost":
        object.__setattr__(trainer._returns[0].measures, "violations", False)
    else:
        object.__setattr__(trainer._fitted.objective, "compute_weight", 2.0)
    before = WORK["predicts"], WORK["updates"]
    with pytest.raises(ValueError):
        stream.run_contextual_window(trainer)
    assert (WORK["predicts"], WORK["updates"]) == before


def test_should_detach_application_snapshot_and_preserve_legacy_snapshot_keys():
    stream, source, _, _ = make()
    trainer, _, _, _ = fit_for(stream, source)
    snapshot = trainer.application_snapshot()
    snapshot["source"].state["optimizer"][0] = 123
    snapshot["fit"]["returns"].clear()
    assert len(trainer.application_snapshot()["fit"]["returns"]) == 4
    assert trainer.application_snapshot()["source"].state == source.snapshot_state()
    assert set(trainer.snapshot()) == {
        "attempts",
        "completed",
        "discarded_trials",
        "stopped",
        "rows",
        "fitted",
        "clock",
    }


@pytest.mark.parametrize("fault", ["unfrozen", "stopped", "busy"])
def test_should_require_the_original_completed_available_owner(fault):
    stream, source, _, _ = make()
    trainer, _, _, _ = fit_for(stream, source, freeze=fault != "unfrozen")
    if fault == "stopped":
        trainer._stopped = True
    if fault == "busy":
        assert trainer._gate.acquire(blocking=False)
    try:
        with pytest.raises(ValueError):
            stream.run_contextual_window(trainer)
    finally:
        if fault == "busy":
            trainer._gate.release()
    assert not source.calls


def test_should_refuse_bare_fitted_policy_and_arbitrary_callbacks():
    stream, source, _, _ = make()
    trainer, _, _, _ = fit_for(stream, source)
    for value in (trainer.snapshot()["fitted"], lambda observation: "wake1", object()):
        with pytest.raises(ValueError):
            stream.run_contextual_window(value)
        with pytest.raises(ValueError):
            stream.run_reference_window(value)
    assert not source.calls


def test_should_reject_final_or_probe_data_without_accessing_it():
    stream, source, _, _ = make(final=True)
    trainer, _, _, _ = fit_for(stream, source)
    with pytest.raises(ValueError):
        stream.run_contextual_window(trainer)
    assert not source.calls
    for role in ("probe_train", "probe_test", "final_test"):
        with pytest.raises(ValueError):
            AttributionStream(
                LiteralLearner(),
                (AttributionBatch(role, "A", ("forbidden",), np.zeros((1, 1)), np.zeros((1, 1))),),
                (),
                checkpoint_updates=(0, 1),
                label_delay=1,
                budget=ToyBudgetSession(ToyExecutionBudget(max_training_updates=1), lambda: 0.0),
                training_only=True,
            ).run_contextual_window(trainer)


@pytest.mark.parametrize("fault", ["fit", "version", "budget", "lost", "late"])
def test_should_stop_after_one_paid_update_without_repairing_corrupt_or_lost_history(fault):
    stream, source, budget, now = make()
    trainer, current, _, _ = fit_for(stream, source)

    def change():
        if fault == "fit":
            object.__setattr__(trainer._returns[0].measures, "execution_seconds", 2.0)
        elif fault == "version":
            current[0] = replace(current[0], actor_version="changed")
        elif fault == "budget":
            object.__setattr__(budget.budget, "max_training_updates", 5)
        elif fault == "lost":
            raise RuntimeError("lost literal update return")
        else:
            now[0] = 61.0

    source.hooks["train"] = change
    with pytest.raises((ValueError, RuntimeError)):
        stream.run_contextual_window(trainer)
    report = stream.reference_report
    assert report["status"] == "failed" and report["attempted_wakes"] == 1
    assert (
        report["controller_application"]["fit"]["returns"][0]["measures"]["execution_seconds"]
        == 1.0
    )
    assert len([c for c in source.calls if c[0] == "train"]) == 1
    if fault in ("fit", "version"):
        assert budget.updates_completed == len(report["applied_updates"]) == 1
    with pytest.raises(ValueError):
        stream.run_contextual_window(trainer)


def test_should_refuse_reentrant_stream_and_owner_calls_at_source_callback_boundary():
    stream, source, _, _ = make()
    trainer, _, _, hooks = fit_for(stream, source)

    def reenter():
        with pytest.raises(ValueError, match="busy"):
            stream.run_contextual_window(trainer)
        with pytest.raises(ValueError, match="busy"):
            trainer.application_snapshot()

    hooks.append(reenter)
    assert stream.run_contextual_window(trainer)["window_complete"]


def test_should_seal_partial_window_and_leave_fixed_references_unchanged():
    stream, source, _, _ = make()
    trainer, _, _, _ = fit_for(stream, source, mistakes="wait")
    result = stream.run_contextual_window(trainer)
    result["controller_application"]["fit"]["returns"].clear()
    assert len(stream.reference_report["controller_application"]["fit"]["returns"]) == 4
    for operation in (
        lambda: stream.run_contextual_window(trainer),
        stream.finish,
        lambda: stream.run_until(1),
    ):
        with pytest.raises(ValueError):
            operation()
    other, _, _, _ = make()
    fixed = other.run_reference_window(ReferencePolicy("no_learning"))
    assert "controller_application" not in fixed and fixed["policy"]["mode"] == "no_learning"


@pytest.mark.parametrize("fault", ["fit", "return_identity", "learner_source"])
def test_should_refuse_changes_during_diagnostics_before_the_first_wake(fault):
    stream, source, budget, _ = make()
    trainer, _, _, _ = fit_for(stream, source, digest=lambda state: "constant")

    def change():
        if fault == "fit":
            object.__setattr__(trainer._fitted, "choices", deepcopy(trainer._fitted.choices))
        elif fault == "return_identity":
            object.__setattr__(
                trainer._returns[0], "request", deepcopy(trainer._returns[0].request)
            )
        else:
            source.state["optimizer"][0] += 1

    source.hooks["predict"] = change
    before = WORK["updates"]
    with pytest.raises(ValueError):
        stream.run_contextual_window(trainer)
    assert WORK["updates"] == before and budget.updates_completed == 0
    assert stream.reference_report["ready_feedback"]["prediction_calls_attempted"] == 1


def test_should_refuse_a_fit_that_is_future_on_its_original_clock(monkeypatch):
    stream, source, _, _ = make()
    trainer, _, clock, _ = fit_for(stream, source)
    monkeypatch.setattr(clock, "now", lambda: 0)
    with pytest.raises(ValueError, match="not arrived"):
        stream.run_contextual_window(trainer)
    assert not source.calls


@pytest.mark.parametrize("when", ["before", "during"])
def test_should_charge_binding_reads_to_the_original_application_budget(monkeypatch, when):
    stream, source, budget, now = make()
    trainer, _, _, hooks = fit_for(stream, source)
    if when == "before":
        now[0] = 61.0
        monkeypatch.setattr(
            trainer, "application_snapshot", lambda: pytest.fail("expired binding read")
        )
    else:
        hooks.append(lambda: now.__setitem__(0, 61.0))
    with pytest.raises(RuntimeError):
        stream.run_contextual_window(trainer)
    assert budget.started_at == 0.0 and budget.updates_completed == 0 and not source.calls


@pytest.mark.parametrize("fault", ["clock", "candidate"])
def test_should_refuse_callback_state_changes_after_diagnostics_before_any_wake(fault):
    stream, source, budget, _ = make()
    trainer, _, _, hooks = fit_for(stream, source)

    def arm():
        def change():
            if fault == "clock":
                stream._clock.advance_to(4)
            else:
                stream._runtime._candidate.state["optimizer"][0] += 99

        hooks.append(change)

    source.hooks["predict"] = arm
    before = WORK["updates"]
    with pytest.raises(ValueError, match="clock or candidate"):
        stream.run_contextual_window(trainer)
    assert WORK["updates"] == before and budget.updates_completed == 0
    assert stream.reference_report["ready_feedback"]["prediction_calls_completed"] == 1
