"""Opt-in probe diagnostics reuse existing calls and preserve reserved row meaning."""

from copy import deepcopy
from dataclasses import replace
import pickle

import numpy as np
import pytest

import src.app.fresh_task_probe as probe_module
from src.app.toy_execution_budget import ToyBudgetSession
from test_categorical_fresh_task_probe import (
    batch as categorical_batch,
    batches as categorical_batches,
    learner as categorical_learner,
    run as categorical_run,
)
from test_fresh_task_probe import FakeLearner, batches, run, session


def strip_diagnostics(report):
    result = deepcopy(report)
    result.pop("decision_diagnostics_enabled")
    result.pop("decision_diagnostics_scope")
    for checkpoint in result["checkpoints"]:
        checkpoint.pop("decision_diagnostics")
    return result


def test_should_keep_binary_default_and_explicit_false_wire_exact():
    ordinary, disabled = run(FakeLearner()), run(FakeLearner(), decision_diagnostics=False)
    assert pickle.dumps(ordinary) == pickle.dumps(disabled)
    assert "decision_diagnostics_enabled" not in ordinary
    assert all("decision_diagnostics" not in point for point in ordinary["checkpoints"])


@pytest.mark.parametrize("categorical", [False, True])
def test_should_only_add_diagnostic_fields_without_more_native_budget_or_timer_calls(
    monkeypatch, categorical
):
    original_final, original_update, original_record = (
        ToyBudgetSession.before_final,
        ToyBudgetSession.before_update,
        ToyBudgetSession.record_update,
    )
    counts: dict[str, int] = {}

    def before_final(owner):
        counts["before_final"] += 1
        return original_final(owner)

    def before_update(owner):
        counts["before_update"] += 1
        return original_update(owner)

    def record_update(owner):
        counts["record_update"] += 1
        return original_record(owner)

    monkeypatch.setattr(ToyBudgetSession, "before_final", before_final)
    monkeypatch.setattr(ToyBudgetSession, "before_update", before_update)
    monkeypatch.setattr(ToyBudgetSession, "record_update", record_update)
    results, traces, callbacks = [], [], []
    for enabled in (False, True):
        counts = dict(before_final=0, before_update=0, record_update=0, timer=0)
        source = categorical_learner() if categorical else FakeLearner()
        budget = session()

        def timer():
            counts["timer"] += 1
            return 2.5

        selected = categorical_run if categorical else run
        results.append(selected(source, budget=budget, timer=timer, decision_diagnostics=enabled))
        assert budget.updates_completed == 2
        traces.append(pickle.dumps(source.trace))
        callbacks.append(counts.copy())
    assert results[1]["status"] == "complete" and results[1]["decision_diagnostics_enabled"] is True
    assert strip_diagnostics(results[1]) == results[0]
    assert traces[0] == traces[1] and callbacks[0] == callbacks[1]
    assert results[1]["decision_diagnostics_scope"]


def test_should_preserve_binary_soft_target_and_inclusive_threshold_semantics_per_row():
    training, evaluation = batches()
    values = np.array([[0.5], [0.5], [1.0], [0.0]])
    targets = np.array([[0.5], [0.49], [0.0], [1.0]])
    evaluation = (
        replace(
            evaluation[0],
            row_ids=tuple(f"soft-{index}" for index in range(4)),
            features=np.zeros((4, 2)),
            targets=targets,
        ),
    )
    result = run(
        FakeLearner(hooks={"predict": lambda owner, features: values.copy()}),
        training=training,
        evaluation=evaluation,
        decision_diagnostics=True,
    )
    assert result["status"] == "complete"
    clipped = np.clip(values[:, 0], 1e-12, 1 - 1e-12)
    losses = -targets[:, 0] * np.log(clipped) - (1 - targets[:, 0]) * np.log1p(-clipped)
    for point in result["checkpoints"]:
        diagnostic = point["decision_diagnostics"]["C"]
        assert diagnostic["loss_definition"] == "binary_cross_entropy"
        assert diagnostic["target_class_rule"] == "threshold_ge_half"
        assert diagnostic["class_counts"] == [2, 2]
        assert diagnostic["predicted_class_counts"] == [1, 3]
        assert diagnostic["confusion"] == [[0, 2], [1, 1]]
        assert diagnostic["row_ids"] == [f"soft-{index}" for index in range(4)]
        assert diagnostic["episode_id"] is None and diagnostic["sequence_kind"] == "batch_row_order"
        assert [row["predicted_class"] for row in diagnostic["decisions"]] == [1, 1, 1, 0]
        assert [row["target_class"] for row in diagnostic["decisions"]] == [1, 0, 0, 1]
        assert [row["target"] for row in diagnostic["decisions"]] == [0.5, 0.49, 0.0, 1.0]
        np.testing.assert_allclose(
            [row["loss"] for row in diagnostic["decisions"]], losses, rtol=0, atol=1e-12
        )
        assert point["scores"]["C"]["bce"] == pytest.approx(float(np.mean(losses)))


def test_should_bind_categorical_ties_confusion_absent_classes_and_complete_episode_rows():
    training = tuple(categorical_batch(f"diag-train-{index}", classes=4) for index in range(2))
    evaluation = (categorical_batch("diag-eval", "probe_test", classes=4),)
    probabilities = np.array([[0.5, 0.5, 0.0, 0.0], [0.0, 0.5, 0.5, 0.0], [0.0, 0.7, 0.3, 0.0]])
    result = categorical_run(
        categorical_learner(predict=lambda owner, features: probabilities.copy()),
        training=training,
        evaluation=evaluation,
        decision_diagnostics=True,
    )
    assert result["status"] == "complete" and result["reserved_data_bytes"] == 216
    for point in result["checkpoints"]:
        diagnostic = point["decision_diagnostics"]["C"]
        assert diagnostic["num_classes"] == 4 and diagnostic["rows"] == 3
        assert diagnostic["loss_definition"] == "categorical_cross_entropy"
        assert diagnostic["target_class_rule"] == "hard_class_index"
        assert diagnostic["row_ids"] == list(evaluation[0].row_ids)
        assert diagnostic["episode_id"] == "diag-eval"
        assert diagnostic["sequence_kind"] == "complete_episode"
        assert diagnostic["class_counts"] == [1, 1, 1, 0]
        assert diagnostic["predicted_class_counts"] == [1, 2, 0, 0]
        assert diagnostic["confusion"] == [[1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]]
        assert [item["predicted_class"] for item in diagnostic["decisions"]] == [0, 1, 1]
        assert diagnostic["per_class"][3] == dict(
            class_index=3, support=0, predicted_count=0, correct=0, accuracy=None, mean_loss=None
        )
        assert diagnostic["per_class"][2]["accuracy"] == 0.0
        assert diagnostic["per_class"][2]["mean_loss"] == pytest.approx(-np.log(0.3))
        assert point["scores"]["C"]["accuracy"] == pytest.approx(2 / 3)


def test_should_detach_all_diagnostic_payloads_from_predictions_data_source_and_other_checkpoints():
    training, evaluation = categorical_batches()
    predictions = np.tile(np.array([[0.2, 0.3, 0.5]]), (3, 1))
    source = categorical_learner(predict=lambda owner, features: predictions)
    before = pickle.dumps((training, evaluation, source.value, source.policy))
    result = categorical_run(
        source, training=training, evaluation=evaluation, decision_diagnostics=True
    )
    saved = deepcopy(result)
    predictions.fill(999)
    assert result == saved
    first = result["checkpoints"][0]["decision_diagnostics"]["C"]
    second = result["checkpoints"][1]["decision_diagnostics"]["C"]
    first["confusion"][0][0] = 999
    first["row_ids"][0] = "changed"
    first["decisions"][0]["target"] = 999
    assert second == saved["checkpoints"][1]["decision_diagnostics"]["C"]
    assert pickle.dumps((training, evaluation, source.value, source.policy)) == before


def test_should_bind_diagnostic_ids_to_owned_data_before_a_fork_mutates_caller_metadata():
    training, evaluation = categorical_batches()

    def fork():
        object.__setattr__(evaluation[0], "row_ids", ("changed-0", "changed-1", "changed-2"))
        object.__setattr__(evaluation[0].episode, "episode_id", "changed-episode")

    result = categorical_run(
        categorical_learner(fork=fork),
        training=training,
        evaluation=evaluation,
        decision_diagnostics=True,
    )
    assert result["status"] == "complete"
    for point in result["checkpoints"]:
        diagnostic = point["decision_diagnostics"]["C"]
        assert diagnostic["row_ids"] == [f"probe-eval-0:step:{index}" for index in range(3)]
        assert diagnostic["episode_id"] == "probe-eval-0"


@pytest.mark.parametrize("flag", [0, 1, None, "true", np.bool_(True)])
def test_should_refuse_non_bool_flag_before_fork_budget_or_timer_callbacks(monkeypatch, flag):
    source, budget = FakeLearner(), session()

    def forbidden(*args, **kwargs):
        pytest.fail("invalid diagnostic flag reached a callback")

    monkeypatch.setattr(ToyBudgetSession, "before_final", forbidden)
    monkeypatch.setattr(ToyBudgetSession, "before_update", forbidden)
    with pytest.raises(ValueError):
        run(source, budget=budget, timer=forbidden, decision_diagnostics=flag)
    assert source.trace == [] and budget.updates_completed == 0


@pytest.mark.parametrize("fail_at", [1, 4])
def test_should_retain_actual_forward_cost_and_only_full_checkpoints_when_diagnostic_helper_fails(
    monkeypatch, fail_at
):
    training, evaluation = batches()
    second = replace(
        evaluation[0],
        context="D",
        row_ids=("D-0", "D-1"),
        features=evaluation[0].features.copy(),
        targets=evaluation[0].targets.copy(),
    )
    calls: list[None] = []
    original = probe_module.decision_class_diagnostics

    def fail(predictions, targets, num_classes):
        calls.append(None)
        if len(calls) == fail_at:
            raise RuntimeError("derived decision diagnostic fault")
        return original(predictions, targets, num_classes)

    monkeypatch.setattr(probe_module, "decision_class_diagnostics", fail)
    source, budget = FakeLearner(), session()
    result = run(
        source,
        training=training,
        evaluation=(*evaluation, second),
        budget=budget,
        decision_diagnostics=True,
    )
    assert result["status"] == "failed" and "derived decision diagnostic fault" in result["reason"]
    assert (
        result["score_forward_calls_attempted"]
        == result["score_forward_calls_completed"]
        == fail_at
    )
    assert [point["updates"] for point in result["checkpoints"]] == ([] if fail_at == 1 else [0])
    assert result["completed_updates"] == budget.updates_completed == (0 if fail_at == 1 else 1)
    assert len([item for item in source.trace if item[0] == "predict"]) == fail_at
    assert len([item for item in source.trace if item[0] == "train"]) == (0 if fail_at == 1 else 1)


@pytest.mark.parametrize("bad_role", ["final_test", "probe_train"])
def test_should_preserve_reserved_role_gates_when_diagnostics_are_enabled(bad_role):
    training, evaluation = batches()
    if bad_role == "final_test":
        training = (replace(training[0], role="final_test"), training[1])
    else:
        evaluation = (replace(evaluation[0], role="probe_train"),)
    source = FakeLearner()
    with pytest.raises(ValueError):
        run(source, training=training, evaluation=evaluation, decision_diagnostics=True)
    assert source.trace == []
