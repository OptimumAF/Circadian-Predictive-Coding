"""Literal stream prequential decisions, costs and isolation; no actual fitting."""

from copy import deepcopy

import numpy as np
import pytest

from src.app.coupled_attribution import AttributionBatch, AttributionStream
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget, ToyExecutionStopped
from src.core.learner_ports import TrainingDiagnostic
from src.core.reference_controller import ReferencePolicy
from src.app.reference_prequential import ReferencePrequential
from types import SimpleNamespace

WORK = dict(train_calls=0, train_rows=0, predict_calls=0, predict_rows=0)


class Learner:
    def __init__(self, state=0, hooks=None, calls=None):
        self.state = state
        self.hooks = {} if hooks is None else hooks
        self.calls = [] if calls is None else calls

    def snapshot_state(self):
        return [self.state]

    def restore_state(self, state):
        self.state = state[0]
        if "restore" in self.hooks:
            self.hooks["restore"](self)

    def fork(self):
        if "fork" in self.hooks:
            return self.hooks["fork"](self)
        return Learner(self.state, self.hooks, self.calls)

    def train_batch(self, features, targets):
        WORK["train_calls"] += 1
        WORK["train_rows"] += len(features)
        self.calls.append(("train", int(features[0, 0]), self.state))
        self.state += 1
        return TrainingDiagnostic("literal", 0.0)

    def predict(self, features):
        WORK["predict_calls"] += 1
        WORK["predict_rows"] += len(features)
        assert type(features) is np.ndarray and features.shape == (1, 1)
        self.calls.append(("predict", int(features[0, 0]), self.state))
        if "predict" in self.hooks:
            return self.hooks["predict"](self, features)
        return np.zeros((len(features), 1), dtype=np.float64)


def make(*, delay=1, target=1.0, classes=2, rows=4, credit=4):
    source = Learner()
    now = [0.0]
    budget = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=credit, max_wall_seconds=60), lambda: now[0]
    )
    batches = tuple(
        AttributionBatch(
            "train",
            "A",
            (f"row-{i}",),
            np.array([[float(i)]]),
            np.array([[target]]),
            num_classes=classes,
        )
        for i in range(rows)
    )
    stream = AttributionStream(
        source,
        batches,
        (),
        checkpoint_updates=tuple(range(rows + 1)),
        label_delay=delay,
        budget=budget,
        training_only=True,
    )
    return stream, source, budget, now


def run(owner, mode="drift"):
    return owner.run_reference_window(ReferencePolicy(mode), prequential_feedback=True)


def test_should_predict_before_each_label_and_update_then_retire_stale_feedback():
    owner, source, budget, _ = make()
    source_before = deepcopy(source.snapshot_state())
    actor_before = owner._runtime.actor.snapshot_state()
    result = run(owner)
    assert source.calls == [
        ("predict", 0, 0),
        ("predict", 1, 0),
        ("train", 0, 0),
        ("predict", 2, 1),
        ("predict", 3, 1),
        ("train", 1, 1),
    ]
    p = result["prequential"]
    assert [e["eligible"] for e in p["events"]] == [True, False, True, False]
    assert [(e["observed_at"], e["arrived_at"]) for e in p["events"]] == [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
    ]
    assert [e["reason"] for e in p["events"]] == ["current_candidate", "stale_candidate"] * 2
    assert [r["applied_at"] for r in result["applied_updates"]] == [1, 3]
    assert result["decisions"][2]["observation"]["feedback"] is None
    assert (
        result["window_complete"]
        and not result["training_complete"]
        and result["pending_unapplied"] == 2
    )
    assert budget.updates_completed == 2 and result["score_forward_calls"] == 0
    assert (
        p["fork_attempts"]
        == p["fork_completed"]
        == p["prediction_calls_attempted"]
        == p["prediction_calls_completed"]
        == 4
    )
    assert p["prediction_rows_attempted"] == p["prediction_rows_completed"] == 4
    assert p["pending_packets"] == 0 and p["peak_pending_rows"] == 2
    assert (
        source.snapshot_state() == source_before
        and owner._runtime.actor.snapshot_state() == actor_before
    )


@pytest.mark.parametrize("mode", ["no_learning", "periodic", "budget_aware", "drift"])
def test_should_observe_identical_packet_count_for_each_opted_in_control(mode):
    owner, source, _, _ = make()
    result = run(owner, mode)
    assert result["prequential"]["prediction_calls_completed"] == 4
    assert [i for kind, i, _ in source.calls if kind == "predict"] == [0, 1, 2, 3]
    assert len(result["prequential"]["events"]) == 4


def test_should_wait_for_correct_predictions_without_fabricated_alarm():
    owner, _, budget, _ = make(target=0.0)
    result = run(owner)
    assert budget.updates_completed == 0
    assert all(e["mistakes"] == 0 and e["eligible"] for e in result["prequential"]["events"])


@pytest.mark.parametrize("delay,ticks", [(0, [0, 1, 2, 3]), (2, [2, 5])])
def test_should_use_original_zero_and_delayed_label_clocks_including_tail(delay, ticks):
    owner, _, _, _ = make(delay=delay)
    result = run(owner)
    assert [r["applied_at"] for r in result["applied_updates"]] == ticks
    assert [e["arrived_at"] for e in result["prequential"]["events"]] == list(
        range(delay, delay + 4)
    )


def test_should_support_explicit_categorical_probabilities_and_original_tie_direction():
    owner, source, _, _ = make(classes=3)
    source.hooks["predict"] = lambda model, features: np.array([[0.5, 0.5, 0.0]])
    result = run(owner)
    assert [e["mistakes"] for e in result["prequential"]["events"]] == [1] * 4


@pytest.mark.parametrize("value", [None, 1, "yes"])
def test_should_refuse_nonboolean_feedback_flag_before_work(value):
    owner, source, budget, _ = make()
    with pytest.raises(ValueError):
        owner.run_reference_window(ReferencePolicy("drift"), prequential_feedback=value)
    assert owner.next_tick == 0 and not source.calls and budget.updates_completed == 0


def test_should_refuse_soft_binary_targets_before_prediction_or_tick():
    owner, source, _, _ = make(target=0.5)
    with pytest.raises(ValueError, match="hard-class"):
        run(owner)
    assert not source.calls and owner.next_tick == 0 and owner.reference_report is None


@pytest.mark.parametrize(
    "bad",
    [
        np.array([[np.nan]]),
        np.array([[1.1]]),
        np.zeros((2, 1)),
        np.zeros((1, 1), dtype=np.float32),
        [[0.0]],
    ],
)
def test_should_preserve_paid_prediction_and_refuse_malformed_results_without_training(bad):
    owner, source, budget, _ = make()
    source.hooks["predict"] = lambda model, features: bad
    with pytest.raises(ValueError):
        run(owner)
    result = owner.reference_report
    assert result["status"] == "failed" and budget.updates_completed == 0
    assert result["prequential"]["prediction_calls_completed"] == 1
    assert result["next_tick"] == 0 and len(source.calls) == 1


def test_should_report_unreturned_prediction_attempt_without_retry():
    owner, source, _, _ = make()

    def fail(model, features):
        raise RuntimeError("predict failed")

    source.hooks["predict"] = fail
    with pytest.raises(RuntimeError, match="predict failed"):
        run(owner)
    report = owner.reference_report["prequential"]
    assert report["prediction_calls_attempted"] == 1 and report["prediction_calls_completed"] == 0
    assert report["unreturned_prediction_unknown"] and not report["unreturned_fork_unknown"]
    with pytest.raises(ValueError, match="failed or finished"):
        run(owner)


def test_should_charge_late_prediction_to_original_wall_budget_and_stop_before_labels():
    owner, source, budget, now = make()

    def late(model, features):
        now[0] = 61.0
        return np.zeros((1, 1))

    source.hooks["predict"] = late
    with pytest.raises(ToyExecutionStopped):
        run(owner)
    assert budget.updates_completed == 0 and len(source.calls) == 1
    assert owner.reference_report["prequential"]["prediction_calls_completed"] == 1


@pytest.mark.parametrize("kind", ["source", "actor", "candidate", "predictor"])
def test_should_refuse_prediction_state_mutations_before_more_work(kind):
    owner, source, budget, _ = make()

    def mutate(model, features):
        targets = dict(
            source=source,
            actor=owner._runtime.actor._learner,
            candidate=owner._runtime._candidate,
            predictor=model,
        )
        targets[kind].state += 10
        return np.zeros((1, 1))

    source.hooks["predict"] = mutate
    with pytest.raises(ValueError):
        run(owner)
    assert budget.updates_completed == 0 and len(source.calls) == 1


def test_should_refuse_aliasing_fork_before_prediction():
    owner, source, _, _ = make()
    source.hooks["fork"] = lambda model: source
    with pytest.raises(ValueError, match="independently"):
        run(owner)
    assert not source.calls


def test_should_refuse_fork_restore_mismatch_before_prediction():
    owner, source, _, _ = make()
    source.hooks["restore"] = lambda model: setattr(model, "state", 99)
    with pytest.raises(ValueError, match="captured candidate"):
        run(owner)
    assert not source.calls


def test_should_return_detached_scalar_evidence_without_retaining_prediction_chain():
    owner, _, _, _ = make()
    result = run(owner)
    result["prequential"]["events"].clear()
    assert len(owner.reference_report["prequential"]["events"]) == 4
    assert not hasattr(owner, "_reference_prequential")


def test_should_keep_original_no_prediction_default_report():
    owner, source, _, _ = make()
    result = owner.run_reference_window(ReferencePolicy("no_learning"))
    assert "prequential" not in result and not source.calls


@pytest.mark.parametrize("role", ["final_test", "probe_train"])
def test_should_refuse_nontrain_helper_inputs_without_predictions(role):
    batch = SimpleNamespace(
        role=role, episode=None, num_classes=2, features=np.zeros((1, 1)), targets=np.zeros((1, 1))
    )
    with pytest.raises(ValueError, match="TRAIN"):
        ReferencePrequential((batch,))


def test_should_refuse_pending_storage_overflow_before_any_tick_or_fork():
    source = Learner()
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=5), lambda: 0.0)
    batches = tuple(
        AttributionBatch(
            "train",
            "A",
            tuple(f"r-{i}-{j}" for j in range(1024)),
            np.zeros((1024, 1)),
            np.zeros((1024, 1)),
        )
        for i in range(5)
    )
    owner = AttributionStream(
        source,
        batches,
        (),
        checkpoint_updates=tuple(range(6)),
        label_delay=1,
        budget=budget,
        training_only=True,
    )
    with pytest.raises(ValueError, match="bounded"):
        run(owner)
    assert not source.calls and owner.next_tick == 0 and owner.reference_report is None


def test_should_preserve_failed_fork_attempt_without_prediction_or_training():
    owner, source, _, _ = make()

    def fail(model):
        raise RuntimeError("fork failed")

    source.hooks["fork"] = fail
    with pytest.raises(RuntimeError, match="fork failed"):
        run(owner)
    report = owner.reference_report["prequential"]
    assert report["fork_attempts"] == 1 and report["fork_completed"] == 0
    assert report["unreturned_fork_unknown"] and not report["unreturned_prediction_unknown"]
    assert not source.calls


def test_should_refuse_late_fork_before_restore_or_prediction():
    owner, source, budget, now = make()

    def late(model):
        now[0] = 61.0
        return Learner(0, source.hooks, source.calls)

    source.hooks["fork"] = late
    with pytest.raises(ToyExecutionStopped):
        run(owner)
    report = owner.reference_report["prequential"]
    assert report["fork_completed"] == 1 and report["prediction_calls_attempted"] == 0
    assert not source.calls and budget.updates_completed == 0
