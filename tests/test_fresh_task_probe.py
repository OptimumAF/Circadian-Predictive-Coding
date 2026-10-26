"""Disposable fake-learner probes test isolation and honest partial accounting."""

from copy import deepcopy
from dataclasses import replace
import pickle
from typing import Any

import numpy as np
import pytest

import src.app.fresh_task_probe as probe_module
from src.app.coupled_attribution import AttributionBatch
from src.app.fresh_task_probe import run_fresh_task_probe
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.learner_ports import TrainingDiagnostic


class FakeLearner:
    def __init__(self, *, trace=None, hooks=None, same=False, value=99.0):
        self.trace = [] if trace is None else trace
        self.hooks = {} if hooks is None else hooks
        self.same = same
        self.value = np.array([value], dtype=np.float64)
        self.policy = {"learning_rate": 0.01}
        self.child: Any = None

    def fork(self):
        self.trace.append(("fork",))
        if "fork" in self.hooks:
            self.hooks["fork"]()
        if self.same:
            return self
        self.child = FakeLearner(trace=self.trace, hooks=self.hooks, value=float(self.value[0]))
        self.child.policy = deepcopy(self.policy)
        return self.child

    def restore_state(self, state):
        self.trace.append(("restore", state.copy()))
        self.value = state.copy()
        if "restore" in self.hooks:
            self.hooks["restore"](self, state)

    def train_batch(self, features, targets):
        self.trace.append(("train", float(self.value[0]), features.copy(), targets.copy()))
        self.value += 1
        if "train" in self.hooks:
            result = self.hooks["train"](self, features, targets)
            if result is not None:
                return result
        return TrainingDiagnostic("engineering_fake", float(self.value[0]))

    def predict(self, features):
        self.trace.append(("predict", float(self.value[0]), features.copy()))
        if "predict" in self.hooks:
            return self.hooks["predict"](self, features)
        return np.full((len(features), 1), (float(self.value[0]) + 1) / 4, dtype=np.float64)

    def snapshot_state(self):
        pytest.fail("probe requested a source or disposable native snapshot")


def batches():
    training = tuple(
        AttributionBatch(
            "probe_train",
            "C",
            (f"train-{number}",),
            np.array([[0.1 + number / 10, 0.2]], dtype=np.float64),
            np.array([[1.0]]),
        )
        for number in range(2)
    )
    evaluation = (
        AttributionBatch(
            "probe_test",
            "C",
            ("test-0", "test-1"),
            np.array([[-0.1, 0.2], [0.3, -0.4]]),
            np.ones((2, 1), dtype=np.float64),
        ),
    )
    return training, evaluation


def session(updates=2, seconds=None):
    return ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=updates, max_wall_seconds=1.0),
        lambda: 0.0 if seconds is None else seconds[0],
    )


def run(source, *, state=None, training=None, evaluation=None, budget=None, **kwargs):
    default_training, default_evaluation = batches()
    return run_fresh_task_probe(
        source,
        np.array([0.0]) if state is None else state,
        default_training if training is None else training,
        default_evaluation if evaluation is None else evaluation,
        budget=session() if budget is None else budget,
        checkpoint_updates=kwargs.pop("checkpoint_updates", (0, 1, 2)),
        timer=kwargs.pop("timer", lambda: 2.5),
        **kwargs,
    )


def test_should_score_fixed_learning_checkpoints_and_keep_benchmark_state_and_data_unchanged():
    source, state = FakeLearner(), np.array([0.0])
    training, evaluation = batches()
    original = pickle.dumps((source.value, source.policy, state, training, evaluation))
    budget = session()
    report = run(source, state=state, training=training, evaluation=evaluation, budget=budget)
    assert report["status"] == "complete" and report["reason"] is None
    assert (
        report["attempted_updates"] == report["completed_updates"] == budget.updates_completed == 2
    )
    assert report["attempted_training_examples"] == report["completed_training_examples"] == 2
    assert report["score_forward_calls_attempted"] == report["score_forward_calls_completed"] == 3
    assert report["elapsed_seconds"] == 0.0 and report["timing_status"] == "complete"
    assert report["reserved_data_bytes"] == 96
    assert [item["updates"] for item in report["checkpoints"]] == [0, 1, 2]
    for index, checkpoint in enumerate(report["checkpoints"]):
        score = checkpoint["scores"]["C"]
        assert score["bce"] == pytest.approx(-np.log((index + 1) / 4))
        assert score["accuracy"] == (0.0 if index == 0 else 1.0)
    assert report["diagnostics"] == [
        {"definition": "engineering_fake", "value": 1.0},
        {"definition": "engineering_fake", "value": 2.0},
    ]
    assert [event[0] for event in source.trace] == [
        "fork",
        "restore",
        "predict",
        "train",
        "predict",
        "train",
        "predict",
    ]
    assert pickle.dumps((source.value, source.policy, state, training, evaluation)) == original
    assert source.child is not source and source.child.policy is not source.policy


def test_should_score_all_declared_contexts_with_fixed_clipping_and_inclusive_half_threshold():
    training, _ = batches()
    evaluation = tuple(
        AttributionBatch(
            "probe_test",
            context,
            tuple(f"{context}-{n}" for n in range(3)),
            np.ones((3, 2), dtype=np.float64),
            np.array([[1.0], [0.0], [1.0]]),
        )
        for context in ("C", "D")
    )
    source = FakeLearner(
        hooks={"predict": lambda learner, features: np.array([[0.0], [1.0], [0.5]])}
    )
    report = run(source, training=training, evaluation=evaluation, checkpoint_updates=(0, 2))
    p, y = np.clip(np.array([0.0, 1.0, 0.5]), 1e-12, 1 - 1e-12), np.array([1.0, 0.0, 1.0])
    expected = float(np.mean(-y * np.log(p) - (1 - y) * np.log1p(-p)))
    assert report["status"] == "complete" and report["score_forward_calls_completed"] == 4
    for checkpoint in report["checkpoints"]:
        assert set(checkpoint["scores"]) == {"C", "D"}
        for score in checkpoint["scores"].values():
            assert score["bce"] == pytest.approx(expected)
            assert score["accuracy"] == pytest.approx(1 / 3)


def test_should_detach_checkpoint_and_every_native_input_from_reserved_data():
    def restore(learner, value):
        value.fill(77)

    def train(learner, features, targets):
        features.fill(88)
        targets.fill(99)

    def predict(learner, features):
        features.fill(66)
        return np.full((len(features), 1), 0.5)

    source = FakeLearner(hooks={"restore": restore, "train": train, "predict": predict})
    state, (training, evaluation) = np.array([0.0]), batches()
    before = pickle.dumps((state, training, evaluation, source.value, source.policy))
    report = run(source, state=state, training=training, evaluation=evaluation)
    assert report["status"] == "complete"
    assert pickle.dumps((state, training, evaluation, source.value, source.policy)) == before
    for event, batch in zip(
        (item for item in source.trace if item[0] == "train"), training, strict=True
    ):
        np.testing.assert_array_equal(event[2], batch.features)
        np.testing.assert_array_equal(event[3], batch.targets)
    for event in (item for item in source.trace if item[0] == "predict"):
        np.testing.assert_array_equal(event[2], evaluation[0].features)


def test_should_finish_reserved_copies_before_an_external_fork_can_change_caller_inputs():
    state, (training, evaluation) = np.array([0.0]), batches()
    before = tuple(item.features.copy() for item in (*training, *evaluation))

    def fork():
        state.fill(55)
        for batch in (*training, *evaluation):
            batch.features.fill(44)
            batch.targets.fill(33)

    source = FakeLearner(hooks={"fork": fork})
    report = run(source, state=state, training=training, evaluation=evaluation)
    assert report["status"] == "complete" and source.value[0] == 99
    np.testing.assert_array_equal(source.trace[1][1], [0.0])
    trains = [event for event in source.trace if event[0] == "train"]
    for event, expected in zip(trains, before[:2], strict=True):
        np.testing.assert_array_equal(event[2], expected)
        np.testing.assert_array_equal(event[3], [[1.0]])
    for event in (item for item in source.trace if item[0] == "predict"):
        np.testing.assert_array_equal(event[2], before[2])


def test_should_keep_failed_after_mutation_attempts_without_committing_scoring_or_retrying():
    def fail(learner, features, targets):
        raise RuntimeError("injected after fake native mutation")

    source, budget = FakeLearner(hooks={"train": fail}), session()
    report = run(source, budget=budget)
    assert (
        report["status"] == "failed" and "injected after fake native mutation" in report["reason"]
    )
    assert report["attempted_updates"] == report["attempted_training_examples"] == 1
    assert (
        report["completed_updates"]
        == report["completed_training_examples"]
        == budget.updates_completed
        == 0
    )
    assert source.child.value[0] == 1 and source.value[0] == 99
    assert report["diagnostics"] == []
    assert [item["updates"] for item in report["checkpoints"]] == [0]
    assert report["score_forward_calls_attempted"] == report["score_forward_calls_completed"] == 1


def test_should_count_a_returned_commit_before_a_post_update_wall_limit_failure():
    seconds = [0.0]

    def train(learner, features, targets):
        seconds[0] = 1.0

    source, budget = FakeLearner(hooks={"train": train}), session(seconds=seconds)
    report = run(source, budget=budget)
    assert report["status"] == "failed" and "max_wall_seconds" in report["reason"]
    assert (
        report["attempted_updates"] == report["completed_updates"] == budget.updates_completed == 1
    )
    assert report["attempted_training_examples"] == report["completed_training_examples"] == 1
    assert len(report["diagnostics"]) == 1
    assert [item["updates"] for item in report["checkpoints"]] == [0]
    assert report["score_forward_calls_completed"] == 1


def test_should_refuse_a_same_object_fork_without_restore_train_or_prediction():
    source = FakeLearner(same=True)
    report = run(source)
    assert report["status"] == "failed" and "distinct" in report["reason"]
    assert source.trace == [("fork",)] and source.value[0] == 99
    assert report["attempted_updates"] == report["score_forward_calls_attempted"] == 0


@pytest.mark.parametrize("phase", ("fork", "restore"))
def test_should_report_external_fork_or_restore_errors_without_later_operations(phase):
    def fail(*args):
        raise RuntimeError(f"injected {phase}")

    source = FakeLearner(hooks={phase: fail})
    report = run(source)
    assert report["status"] == "failed" and f"injected {phase}" in report["reason"]
    assert report["attempted_updates"] == report["score_forward_calls_attempted"] == 0
    assert [event[0] for event in source.trace] == (
        ["fork"] if phase == "fork" else ["fork", "restore"]
    )


@pytest.mark.parametrize("bad", ("nan", "range", "shape", "dtype", "raises"))
def test_should_retain_prediction_attempts_and_returned_forwards_without_scoring_bad_outputs(bad):
    def predict(learner, features):
        if bad == "raises":
            raise RuntimeError("injected forward failure")
        if bad == "shape":
            return np.zeros((len(features), 2))
        if bad == "dtype":
            return np.zeros((len(features), 1), dtype=np.float32)
        return np.full((len(features), 1), np.nan if bad == "nan" else 1.1)

    report = run(FakeLearner(hooks={"predict": predict}))
    assert report["status"] == "failed" and report["checkpoints"] == []
    assert report["score_forward_calls_attempted"] == 1
    assert report["score_forward_calls_completed"] == (0 if bad == "raises" else 1)
    assert report["attempted_updates"] == report["completed_updates"] == 0


def test_should_not_publish_a_partial_checkpoint_when_a_second_context_fails():
    _, evaluation = batches()
    second = replace(
        evaluation[0],
        context="D",
        row_ids=("d-0", "d-1"),
        features=evaluation[0].features.copy(),
        targets=evaluation[0].targets.copy(),
    )
    forwards = [0]

    def predict(learner, features):
        forwards[0] += 1
        if forwards[0] == 2:
            raise RuntimeError("second context failed")
        return np.full((len(features), 1), 0.5)

    report = run(FakeLearner(hooks={"predict": predict}), evaluation=(*evaluation, second))
    assert report["status"] == "failed" and report["checkpoints"] == []
    assert (
        report["score_forward_calls_attempted"] == 2
        and report["score_forward_calls_completed"] == 1
    )
    assert report["attempted_updates"] == 0


def test_should_count_a_returned_native_call_even_when_its_diagnostic_is_invalid():
    budget = session()
    report = run(FakeLearner(hooks={"train": lambda *args: object()}), budget=budget)
    assert report["status"] == "failed" and "TrainingDiagnostic" in report["reason"]
    assert (
        report["attempted_updates"] == report["completed_updates"] == budget.updates_completed == 1
    )
    assert report["diagnostics"] == [] and report["score_forward_calls_completed"] == 1


class BrokenState:
    def __deepcopy__(self, memo):
        raise RuntimeError("checkpoint copy failed")


def test_should_report_checkpoint_copy_failure_before_any_source_or_budget_work():
    source, budget = FakeLearner(), session()
    report = run(source, state=BrokenState(), budget=budget)
    assert report["status"] == "failed" and "checkpoint copy failed" in report["reason"]
    assert source.trace == [] and budget.updates_completed == 0
    assert report["elapsed_seconds"] is None and report["timing_status"] == "unavailable"
    assert report["attempted_updates"] == report["score_forward_calls_attempted"] == 0


def test_should_report_reserved_batch_copy_failure_without_claiming_partial_owned_data(monkeypatch):
    source, budget = FakeLearner(), session()

    def fail_copy(batch):
        raise RuntimeError("reserved batch copy failed")

    def forbidden(*args, **kwargs):
        pytest.fail("failed batch detachment reached an external operational callback")

    monkeypatch.setattr(probe_module, "_copy_batch", fail_copy)
    monkeypatch.setattr(ToyBudgetSession, "before_final", forbidden)
    monkeypatch.setattr(ToyBudgetSession, "before_update", forbidden)
    report = run(source, budget=budget, timer=forbidden)
    assert report["status"] == "failed" and "reserved batch copy failed" in report["reason"]
    assert report["reserved_data_bytes"] is None
    assert report["elapsed_seconds"] is None and report["timing_status"] == "unavailable"
    assert source.trace == [] and budget.updates_completed == 0
    assert report["attempted_updates"] == report["completed_updates"] == 0
    assert report["score_forward_calls_attempted"] == report["score_forward_calls_completed"] == 0
    assert report["checkpoints"] == report["diagnostics"] == []


@pytest.mark.parametrize("bad", ("nan", "bool", "backwards", "raises"))
def test_should_retain_failed_rows_for_invalid_or_unavailable_monotonic_timing(bad):
    calls = [0]

    def timer():
        calls[0] += 1
        if bad == "raises":
            raise RuntimeError("timer unavailable")
        if bad == "nan":
            return float("nan")
        if bad == "bool":
            return True
        return 1.0 if calls[0] == 1 else 0.0

    report = run(FakeLearner(), timer=timer)
    assert report["status"] == "failed" and report["attempted_updates"] == 0
    assert report["score_forward_calls_attempted"] == 0
    assert report["timing_status"] == ("partial" if bad == "backwards" else "unavailable")
    assert report["elapsed_seconds"] == (0.0 if bad == "backwards" else None)


@pytest.mark.parametrize(
    "bad",
    (
        "train_role",
        "test_role",
        "train_list",
        "empty_train",
        "empty_test",
        "train_count",
        "test_count",
        "duplicate_context",
        "missing_context",
        "duplicate_ids",
        "train_alias",
        "within_batch_alias",
        "dtype",
        "target_nan",
        "target_range",
        "width",
        "rows",
        "target_shape",
    ),
)
def test_should_refuse_invalid_reserved_data_before_any_external_learner_or_budget_call(
    monkeypatch, bad
):
    training, evaluation = batches()
    training, evaluation = list(training), list(evaluation)
    if bad == "train_role":
        training[0] = replace(training[0], role="final_test")
    elif bad == "test_role":
        evaluation[0] = replace(evaluation[0], role="final_test")
    elif bad in ("train_list", "empty_train", "empty_test"):
        pass
    elif bad == "train_count":
        training = [
            replace(
                training[0],
                row_ids=(f"train-{n}",),
                features=training[0].features.copy(),
                targets=training[0].targets.copy(),
            )
            for n in range(17)
        ]
    elif bad == "test_count":
        evaluation = [
            replace(
                evaluation[0],
                context=f"C{n}",
                row_ids=(f"test-{n}-0", f"test-{n}-1"),
                features=evaluation[0].features.copy(),
                targets=evaluation[0].targets.copy(),
            )
            for n in range(5)
        ]
    elif bad == "duplicate_context":
        evaluation.append(
            replace(
                evaluation[0],
                row_ids=("other-0", "other-1"),
                features=evaluation[0].features.copy(),
                targets=evaluation[0].targets.copy(),
            )
        )
    elif bad == "missing_context":
        training[0] = replace(training[0], context="unknown")
    elif bad == "duplicate_ids":
        evaluation[0] = replace(evaluation[0], row_ids=("train-0", "test-1"))
    elif bad == "train_alias":
        training[1] = replace(training[1], features=training[0].features)
    elif bad == "within_batch_alias":
        training[0] = replace(training[0], targets=training[0].features[:, :1])
    elif bad == "dtype":
        training[0] = replace(training[0], features=training[0].features.astype(np.float32))
    elif bad in ("target_nan", "target_range", "target_shape"):
        value = (
            np.array([[np.nan]])
            if bad == "target_nan"
            else np.array([[2.0]])
            if bad == "target_range"
            else np.ones((1, 2))
        )
        training[0] = replace(training[0], targets=value)
    elif bad == "width":
        training[0] = replace(training[0], features=np.ones((1, 3)))
    else:
        training[1] = replace(
            training[1],
            row_ids=tuple(f"row-{n}" for n in range(129)),
            features=np.ones((129, 2)),
            targets=np.zeros((129, 1)),
        )
    train: Any = (
        training if bad == "train_list" else () if bad == "empty_train" else tuple(training)
    )
    tests = () if bad == "empty_test" else tuple(evaluation)
    source, budget = FakeLearner(), session()

    def forbidden(*args, **kwargs):
        pytest.fail("invalid probe input reached an external budget boundary")

    monkeypatch.setattr(ToyBudgetSession, "before_final", forbidden)
    monkeypatch.setattr(ToyBudgetSession, "before_update", forbidden)
    if bad == "rows":
        # An oversized later batch must refuse before even the first finite
        # value scan, as well as before learner and budget callbacks.
        monkeypatch.setattr(probe_module, "_validate_batch", forbidden)
    with pytest.raises(ValueError):
        run(source, training=train, evaluation=tests, budget=budget)
    assert source.trace == [] and budget.updates_completed == 0
    if bad == "train_alias":
        clean_training, clean_evaluation = batches()
        cross = replace(
            clean_evaluation[0],
            row_ids=("cross-test",),
            features=clean_training[0].features,
            targets=np.ones((1, 1)),
        )
        with pytest.raises(ValueError):
            run(source, training=clean_training, evaluation=(cross,), budget=budget)
        assert source.trace == [] and budget.updates_completed == 0


@pytest.mark.parametrize(
    "bad",
    (
        "budget_type",
        "spent",
        "limit",
        "checkpoint_list",
        "checkpoint_bool",
        "checkpoint_start",
        "checkpoint_end",
        "checkpoint_order",
        "timer",
    ),
)
def test_should_refuse_invalid_fixed_controls_before_fork_or_budget_callbacks(monkeypatch, bad):
    source, budget = FakeLearner(), session()
    options: dict[str, Any] = {"budget": budget}
    if bad == "budget_type":
        options["budget"] = object()
    elif bad == "spent":
        budget.record_update()
    elif bad == "limit":
        options["budget"] = session(1)
    elif bad == "timer":
        options["timer"] = None
    else:
        options["checkpoint_updates"] = {
            "checkpoint_list": [0, 2],
            "checkpoint_bool": (0, True, 2),
            "checkpoint_start": (1, 2),
            "checkpoint_end": (0, 1),
            "checkpoint_order": (0, 1, 1, 2),
        }[bad]

    def forbidden(*args, **kwargs):
        pytest.fail("invalid fixed controls reached an external budget boundary")

    monkeypatch.setattr(ToyBudgetSession, "before_final", forbidden)
    monkeypatch.setattr(ToyBudgetSession, "before_update", forbidden)
    with pytest.raises(ValueError):
        run(source, **options)
    assert source.trace == []
