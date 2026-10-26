"""Full-width reserved probes retain the existing disposable-learning envelope."""

from dataclasses import replace
import pickle
from typing import Any

import numpy as np
import pytest

from src.app.coupled_attribution import AttributionBatch
from src.app.fresh_task_probe import run_fresh_task_probe
from src.app.toy_execution_budget import ToyBudgetSession
from src.core.demonstration_episode import (
    DemonstrationEpisode,
    DemonstrationTransition,
    episode_arrays,
)
from test_fresh_task_probe import FakeLearner, session


PROBABILITIES = np.array([[0.5, 0.25, 0.05, 0.05, 0.05, 0.05, 0.05]], dtype=np.float64)


def packet(cue, role):
    def observation(step):
        return (-1.0,) * 147 + (float(2 * (step % 4) / 3 - 1),)

    identity = f"{role}-{cue}"
    episode = DemonstrationEpisode(
        identity,
        f"E-{cue}",
        role,
        tuple(
            DemonstrationTransition(
                index,
                observation(index),
                index % 7,
                0.5 if index == 8 else 0.0,
                observation(index + 1),
                index == 8,
                7,
            )
            for index in range(9)
        ),
        7,
    )
    features, targets = episode_arrays(episode)
    return AttributionBatch(
        role,
        episode.context,
        tuple(f"{identity}:step:{index}" for index in range(9)),
        features,
        targets,
        episode,
        7,
    )


def packets():
    return (
        tuple(packet(cue, "probe_train") for cue in ("blue", "red")),
        tuple(packet(cue, "probe_test") for cue in ("blue", "red")),
    )


def plain(identity, role, *, width=148, rows=9, context="E-blue", classes=7):
    return AttributionBatch(
        role,
        context,
        tuple(f"{identity}-{index}" for index in range(rows)),
        np.zeros((rows, width), dtype=np.float64),
        np.zeros((rows, 1), dtype=np.float64),
        num_classes=classes,
    )


def source(hooks=None):
    selected = dict(hooks or {})
    selected.setdefault(
        "predict", lambda learner, features: np.repeat(PROBABILITIES, len(features), axis=0)
    )
    return FakeLearner(hooks=selected)


def run(learner, training, evaluation, *, budget=None, state=None, **kwargs):
    return run_fresh_task_probe(
        learner,
        np.array([0.0]) if state is None else state,
        training,
        evaluation,
        budget=session(updates=len(training)) if budget is None else budget,
        checkpoint_updates=(0, 1, 2) if len(training) == 2 else (0, len(training)),
        timer=lambda: 0.0,
        decision_diagnostics=True,
        **kwargs,
    )


def test_should_probe_complete_nine_step_148_k7_episodes_on_disposable_owner_without_concatenation():
    training, evaluation = packets()
    learner, state, budget = source(), np.array([0.0]), session()
    before = pickle.dumps((learner.value, learner.policy, state, training, evaluation))
    report = run(learner, training, evaluation, state=state, budget=budget)
    assert report["status"] == "complete", report["reason"]
    assert report["num_classes"] == 7 and report["decision_diagnostics_enabled"] is True
    assert (
        report["attempted_updates"] == report["completed_updates"] == budget.updates_completed == 2
    )
    assert report["attempted_training_examples"] == report["completed_training_examples"] == 18
    assert report["score_forward_calls_attempted"] == report["score_forward_calls_completed"] == 6
    assert report["reserved_data_bytes"] == 42912
    assert report["episode_scope"]["training_episode_ids"] == [
        "probe_train-blue",
        "probe_train-red",
    ]
    assert report["episode_scope"]["evaluation_episode_ids"] == [
        "probe_test-blue",
        "probe_test-red",
    ]
    assert report["episode_scope"]["training_transitions"] == 18
    assert report["episode_scope"]["evaluation_transitions"] == 18
    assert [event[0] for event in learner.trace] == [
        "fork",
        "restore",
        "predict",
        "predict",
        "train",
        "predict",
        "predict",
        "train",
        "predict",
        "predict",
    ]
    trains = [event for event in learner.trace if event[0] == "train"]
    assert len(trains) == 2 and all(event[2].shape == (9, 148) for event in trains)
    for event, batch in zip(trains, training, strict=True):
        np.testing.assert_array_equal(event[2], batch.features)
        np.testing.assert_array_equal(event[3], batch.targets)
    targets = np.arange(9) % 7
    expected_ce = float(-np.log(PROBABILITIES[0, targets]).mean())
    for checkpoint in report["checkpoints"]:
        assert set(checkpoint["scores"]) == {"E-blue", "E-red"}
        for cue, batch in zip(("blue", "red"), evaluation, strict=True):
            score = checkpoint["scores"][f"E-{cue}"]
            assert score["ce"] == pytest.approx(expected_ce)
            assert score["accuracy"] == pytest.approx(2 / 9)
            detail = checkpoint["decision_diagnostics"][f"E-{cue}"]
            assert detail["rows"] == len(detail["decisions"]) == 9
            assert detail["sequence_kind"] == "complete_episode"
            assert detail["row_ids"] == list(batch.row_ids)
            assert detail["episode_id"] == batch.episode.episode_id
    assert pickle.dumps((learner.value, learner.policy, state, training, evaluation)) == before
    assert learner.child is not learner and learner.child.policy is not learner.policy
    np.testing.assert_array_equal(learner.value, [99.0])
    np.testing.assert_array_equal(learner.child.value, [2.0])


def test_should_detach_full_width_state_and_native_arrays_before_mutating_callbacks():
    def restore(learner, state):
        state.fill(77)

    def train(learner, features, targets):
        features.fill(88)
        targets.fill(99)

    def predict(learner, features):
        features.fill(66)
        return np.repeat(PROBABILITIES, len(features), axis=0)

    learner = source(dict(restore=restore, train=train, predict=predict))
    training, evaluation = packets()
    state = np.array([0.0])
    before = pickle.dumps((state, training, evaluation, learner.value, learner.policy))
    report = run(learner, training, evaluation, state=state)
    assert report["status"] == "complete" and report["completed_updates"] == 2
    assert pickle.dumps((state, training, evaluation, learner.value, learner.policy)) == before
    assert report["reserved_data_bytes"] == 42912


def test_should_keep_full_width_committed_update_charge_before_post_update_guard_failure():
    learner = source()
    training, evaluation = packets()
    budget = session()

    def guard():
        if learner.child is not None and learner.child.value[0] == 1:
            raise ValueError("benchmark checkpoint changed")

    report = run(learner, training, evaluation, budget=budget, check_boundary=guard)
    assert report["status"] == "failed" and "benchmark checkpoint changed" in report["reason"]
    assert report["completed_updates"] == budget.updates_completed == 1
    assert report["completed_training_examples"] == 9 and len(report["diagnostics"]) == 1
    assert [point["updates"] for point in report["checkpoints"]] == [0]
    assert report["score_forward_calls_completed"] == 2
    assert sum(event[0] == "train" for event in learner.trace) == 1


@pytest.mark.parametrize("width", [1, 16])
def test_should_preserve_original_small_width_probe_admission(width):
    training = (plain("train", "probe_train", width=width),)
    evaluation = (plain("test", "probe_test", width=width),)
    report = run(source(), training, evaluation)
    assert report["status"] == "complete" and report["completed_updates"] == 1


@pytest.mark.parametrize("width", [0, 17, 147, 149])
def test_should_refuse_other_widths_before_any_external_work(monkeypatch, width):
    training = (plain("train", "probe_train", width=width),)
    evaluation = (plain("test", "probe_test", width=width),)
    learner = source()

    def forbidden(*args, **kwargs):
        pytest.fail("invalid width reached external budget callback")

    monkeypatch.setattr(ToyBudgetSession, "before_final", forbidden)
    with pytest.raises(ValueError):
        run(learner, training, evaluation)
    assert learner.trace == []


@pytest.mark.parametrize(
    "fault",
    [
        "mixed_width",
        "aggregate",
        "rows",
        "training_count",
        "evaluation_count",
        "role",
        "duplicate_ids",
        "alias",
        "internal_alias",
        "classes",
        "duplicate_context",
        "missing_context",
        "missing_episode",
    ],
)
def test_should_keep_all_reserved_data_limits_before_fork_or_budget_callbacks(monkeypatch, fault):
    training = [
        plain("train-blue", "probe_train"),
        plain("train-red", "probe_train", context="E-red"),
    ]
    evaluation = [
        plain("test-blue", "probe_test"),
        plain("test-red", "probe_test", context="E-red"),
    ]
    if fault == "mixed_width":
        training[0] = plain("different", "probe_train", width=16)
    elif fault == "aggregate":
        training = [plain(f"large-{index}", "probe_train", rows=128) for index in range(3)]
        evaluation = [plain("large-test", "probe_test", rows=128)]
    elif fault == "rows":
        training[0] = plain("oversized", "probe_train", rows=129)
    elif fault == "training_count":
        training = [plain(f"train-{index}", "probe_train") for index in range(17)]
    elif fault == "evaluation_count":
        evaluation = [
            plain(f"test-{index}", "probe_test", context=f"E{index}") for index in range(5)
        ]
    elif fault == "role":
        training[0] = replace(training[0], role="final_test")
    elif fault == "duplicate_ids":
        evaluation[0] = replace(evaluation[0], row_ids=training[0].row_ids)
    elif fault == "alias":
        evaluation[0] = replace(evaluation[0], features=training[0].features)
    elif fault == "internal_alias":
        training[0] = replace(training[0], targets=training[0].features[:, :1])
    elif fault == "classes":
        training[0] = replace(training[0], num_classes=6)
    elif fault == "duplicate_context":
        evaluation[1] = replace(evaluation[1], context="E-blue")
    elif fault == "missing_context":
        training[0] = replace(training[0], context="undeclared")
    else:
        train, tests = packets()
        training, evaluation = list(train), list(tests)
        training[0] = replace(training[0], episode=None)
    learner, budget = source(), session(updates=len(training))

    def forbidden(*args, **kwargs):
        pytest.fail("invalid reserved data reached an external budget callback")

    monkeypatch.setattr(ToyBudgetSession, "before_final", forbidden)
    monkeypatch.setattr(ToyBudgetSession, "before_update", forbidden)
    with pytest.raises(ValueError):
        run(learner, tuple(training), tuple(evaluation), budget=budget)
    assert learner.trace == [] and budget.updates_completed == 0


@pytest.mark.parametrize("fault", ["spent_budget", "wrong_limit", "bool_checkpoint"])
def test_should_preserve_fresh_budget_and_checkpoint_admission_for_full_width(monkeypatch, fault):
    training, evaluation = packets()
    budget, learner = session(), source()
    points: Any = (0, 1, 2)
    if fault == "spent_budget":
        budget.record_update()
    elif fault == "wrong_limit":
        budget = session(updates=3)
    else:
        points = (0, True, 2)

    def forbidden(*args, **kwargs):
        pytest.fail("invalid controls reached an external budget callback")

    monkeypatch.setattr(ToyBudgetSession, "before_final", forbidden)
    with pytest.raises(ValueError):
        run_fresh_task_probe(
            learner,
            np.array([0.0]),
            training,
            evaluation,
            budget=budget,
            checkpoint_updates=points,
            timer=lambda: 0.0,
        )
    assert learner.trace == []
