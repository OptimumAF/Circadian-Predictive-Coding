"""Reserved categorical episodes stay isolated on disposable fresh-task forks."""

from dataclasses import replace
from math import log
import pickle

import numpy as np
import pytest

import src.app.fresh_task_probe as probe_module
from src.app.coupled_attribution import AttributionBatch, AttributionStream
from src.app.fresh_task_probe import run_fresh_task_probe
from src.core.demonstration_episode import (
    DemonstrationEpisode,
    DemonstrationTransition,
    EpisodeReplaySchema,
    encode_episode,
    episode_arrays,
)
from test_fresh_task_probe import BrokenState, FakeLearner, session


def batch(identity, role="probe_train", *, context="C", classes=3):
    packet = DemonstrationEpisode(
        identity,
        context,
        role,
        tuple(
            DemonstrationTransition(
                index,
                (index / 4, -0.25),
                index,
                float(index == 2),
                ((index + 1) / 4, -0.25),
                index == 2,
                classes,
            )
            for index in range(3)
        ),
        classes,
    )
    features, targets = episode_arrays(packet)
    return AttributionBatch(
        role,
        context,
        tuple(f"{identity}:step:{index}" for index in range(3)),
        features,
        targets,
        packet,
        classes,
    )


def batches(*, metadata=True):
    training = (batch("probe-train-0"), batch("probe-train-1"))
    evaluation = (batch("probe-eval-0", "probe_test"),)
    if not metadata:
        training = tuple(replace(item, episode=None) for item in training)
        evaluation = tuple(replace(item, episode=None) for item in evaluation)
    return training, evaluation


def learner(**hooks):
    return FakeLearner(
        hooks={
            "predict": lambda owner, features: np.tile(
                np.array([[0.2, 0.3, 0.5]]), (len(features), 1)
            ),
            **hooks,
        }
    )


def run(source=None, *, state=None, training=None, evaluation=None, budget=None, **options):
    defaults = batches()
    return run_fresh_task_probe(
        learner() if source is None else source,
        np.array([0.0]) if state is None else state,
        defaults[0] if training is None else training,
        defaults[1] if evaluation is None else evaluation,
        budget=session() if budget is None else budget,
        checkpoint_updates=options.pop("checkpoint_updates", (0, 1, 2)),
        timer=options.pop("timer", lambda: 2.5),
        **options,
    )


def test_should_keep_ordered_complete_episode_calls_and_exact_reserved_scope_costs():
    source, state, budget = learner(), np.array([0.0]), session()
    training, evaluation = batches()
    before = pickle.dumps((state, training, evaluation, source.value, source.policy))
    result = run(source, state=state, training=training, evaluation=evaluation, budget=budget)
    assert result["status"] == "complete" and result["num_classes"] == 3
    assert (
        result["completed_updates"] == result["attempted_updates"] == budget.updates_completed == 2
    )
    assert result["completed_training_examples"] == result["attempted_training_examples"] == 6
    assert result["score_forward_calls_attempted"] == result["score_forward_calls_completed"] == 3
    assert result["reserved_data_bytes"] == 216
    scope = result["episode_scope"]
    assert scope["training_episode_ids"] == ["probe-train-0", "probe-train-1"]
    assert scope["evaluation_episode_ids"] == ["probe-eval-0"]
    assert scope["training_transitions"] == 6 and scope["evaluation_transitions"] == 3
    assert "excluded" in scope["memory_scope"] or "excludes" in scope["memory_scope"]
    calls = [event for event in source.trace if event[0] == "train"]
    assert len(calls) == 2
    for event, packet in zip(calls, training, strict=True):
        np.testing.assert_array_equal(event[2], packet.features)
        np.testing.assert_array_equal(event[3], [[0.0], [1.0], [2.0]])
    for checkpoint in result["checkpoints"]:
        assert checkpoint["scores"]["C"]["ce"] == pytest.approx(
            -(log(0.2) + log(0.3) + log(0.5)) / 3
        )
        assert checkpoint["scores"]["C"]["accuracy"] == pytest.approx(1 / 3)
    assert pickle.dumps((state, training, evaluation, source.value, source.policy)) == before


def test_should_allow_all_metadata_free_categorical_batches_without_episode_scope():
    training, evaluation = batches(metadata=False)
    result = run(training=training, evaluation=evaluation)
    assert result["status"] == "complete" and result["num_classes"] == 3
    assert "episode_scope" not in result


def test_should_apply_manual_categorical_clip_and_first_argmax_tie_without_binary_thresholds():
    training, evaluation = batches(metadata=False)
    evaluation = (replace(evaluation[0], targets=np.array([[1.0], [0.0], [2.0]])),)
    probabilities = np.array([[0.5, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    result = run(
        learner(predict=lambda owner, features: probabilities.copy()),
        training=training,
        evaluation=evaluation,
        checkpoint_updates=(0, 2),
    )
    assert result["status"] == "complete"
    expected = -(log(0.5) + log(1e-12)) / 3
    for checkpoint in result["checkpoints"]:
        assert checkpoint["scores"]["C"] == dict(
            ce=pytest.approx(expected), accuracy=pytest.approx(1 / 3)
        )


@pytest.mark.parametrize(
    "fault",
    [
        "training_count",
        "evaluation_count",
        "soft",
        "dtype",
        "shape",
        "bool_count",
        "episode_count",
        "episode_role",
        "row_ids",
        "features",
        "targets",
        "context",
        "missing_train_metadata",
        "missing_eval_metadata",
        "evaluation_soft",
        "evaluation_shape",
    ],
)
def test_should_refuse_invalid_categorical_or_mixed_episode_ingress_before_callbacks(fault):
    training, evaluation = batches()
    first = training[0]
    assert first.episode is not None
    if fault == "training_count":
        first = batch("probe-train-0", classes=4)
    elif fault == "evaluation_count":
        evaluation = (batch("probe-eval-0", "probe_test", classes=4),)
    elif fault in ("soft", "dtype", "shape", "targets"):
        targets = first.targets.copy()
        if fault == "soft":
            targets[0, 0] = 0.5
        elif fault == "dtype":
            targets = targets.astype(np.float32)
        elif fault == "shape":
            targets = targets[:, 0]
        else:
            targets[0, 0] = 1
        first = replace(first, targets=targets)
    elif fault == "bool_count":
        first = replace(first, num_classes=True)
    elif fault == "episode_count":
        object.__setattr__(first.episode, "num_classes", 4)
    elif fault == "episode_role":
        first = replace(first, episode=replace(first.episode, role="train"))
    elif fault == "row_ids":
        first = replace(first, row_ids=("wrong-0", "wrong-1", "wrong-2"))
    elif fault == "features":
        values = first.features.copy()
        values[0, 0] = 0.125
        first = replace(first, features=values)
    elif fault == "context":
        first = replace(first, context="other")
    elif fault == "missing_train_metadata":
        first = replace(first, episode=None)
    elif fault == "missing_eval_metadata":
        evaluation = (replace(evaluation[0], episode=None),)
    else:
        targets = evaluation[0].targets.copy()
        if fault == "evaluation_soft":
            targets[0, 0] = 0.5
        else:
            targets = targets[:2]
        evaluation = (replace(evaluation[0], targets=targets),)
    source = learner()

    def forbidden():
        pytest.fail("invalid public ingress reached the probe timer")

    with pytest.raises(ValueError):
        run(source, training=(first, *training[1:]), evaluation=evaluation, timer=forbidden)
    assert source.trace == []


@pytest.mark.parametrize("role", ["train", "final_test"])
def test_should_refuse_benchmark_roles_without_automatic_probe_relabeling(role):
    training, evaluation = batches()
    if role == "train":
        training = (batch("benchmark-train", "train"), training[1])
    else:
        evaluation = (batch("benchmark-final", "final_test"),)
    source = learner()
    with pytest.raises(ValueError):
        run(source, training=training, evaluation=evaluation)
    assert source.trace == []


@pytest.mark.parametrize("role", ["probe_train", "probe_test"])
def test_should_refuse_reserved_probe_roles_at_benchmark_stream_ingress(role):
    source = learner()
    with pytest.raises(ValueError):
        AttributionStream(
            source,
            (batch("reserved", role),),
            (batch("benchmark-final", "final_test"),),
            checkpoint_updates=(0, 1),
            label_delay=0,
            budget=session(updates=1),
        )
    assert source.trace == []


@pytest.mark.parametrize("role", ["probe_train", "probe_test"])
def test_should_refuse_reserved_probe_episode_encoding_into_replay(role):
    packet = batch("reserved", role).episode
    assert packet is not None
    with pytest.raises(ValueError):
        encode_episode(packet, EpisodeReplaySchema(2, 3, 3))


def test_should_detach_packet_arrays_and_state_before_fork_mutates_caller_records():
    state, (training, evaluation) = np.array([0.0]), batches()
    expected = [item.features.copy() for item in (*training, *evaluation)]

    def fork():
        state.fill(99)
        for item in (*training, *evaluation):
            item.features.fill(88)
            item.targets.fill(77)
            object.__setattr__(item.episode, "episode_id", "changed")

    source = learner(fork=fork)
    result = run(source, state=state, training=training, evaluation=evaluation)
    assert result["status"] == "complete" and source.value[0] == 99
    assert result["episode_scope"]["training_episode_ids"] == ["probe-train-0", "probe-train-1"]
    np.testing.assert_array_equal(source.trace[1][1], [0.0])
    for call, values in zip(
        (item for item in source.trace if item[0] == "train"), expected[:2], strict=True
    ):
        np.testing.assert_array_equal(call[2], values)
        np.testing.assert_array_equal(call[3], [[0.0], [1.0], [2.0]])


def test_should_detach_each_complete_native_call_and_preserve_original_packets():
    training, evaluation = batches()
    before = pickle.dumps((training, evaluation))

    def train(owner, features, targets):
        features.fill(99)
        targets.fill(99)

    def predict(owner, features):
        features.fill(99)
        return np.tile(np.array([[0.2, 0.3, 0.5]]), (len(features), 1))

    result = run(learner(train=train, predict=predict), training=training, evaluation=evaluation)
    assert result["status"] == "complete"
    assert pickle.dumps((training, evaluation)) == before


@pytest.mark.parametrize("fault", ["shape", "dtype", "sum", "nan", "raises"])
def test_should_preserve_categorical_forward_cost_and_stop_on_invalid_scoring_output(fault):
    def predict(owner, features):
        if fault == "raises":
            raise RuntimeError("categorical prediction fault")
        value = np.tile(np.array([[0.2, 0.3, 0.5]]), (len(features), 1))
        if fault == "shape":
            value = value[:, :1]
        elif fault == "dtype":
            value = value.astype(np.float32)
        elif fault == "sum":
            value[0, 0] = 0
        else:
            value[0, 0] = np.nan
        return value

    result = run(learner(predict=predict))
    assert result["status"] == "failed" and result["checkpoints"] == []
    assert result["score_forward_calls_attempted"] == 1
    assert result["score_forward_calls_completed"] == (0 if fault == "raises" else 1)
    assert result["attempted_updates"] == result["completed_updates"] == 0
    assert result["episode_scope"]["training_transitions"] == 6


def test_should_keep_previous_complete_checkpoint_when_later_second_context_score_fails():
    training, evaluation = batches()
    evaluation = (*evaluation, batch("probe-eval-D", "probe_test", context="D"))
    calls = []

    def predict(owner, features):
        calls.append(features.copy())
        if len(calls) == 4:
            return np.ones((len(features), 1))
        return np.tile(np.array([[0.2, 0.3, 0.5]]), (len(features), 1))

    budget = session()
    result = run(learner(predict=predict), training=training, evaluation=evaluation, budget=budget)
    assert result["status"] == "failed"
    assert [point["updates"] for point in result["checkpoints"]] == [0]
    assert result["completed_updates"] == budget.updates_completed == 1
    assert result["completed_training_examples"] == 3
    assert result["score_forward_calls_attempted"] == result["score_forward_calls_completed"] == 4


def test_should_keep_failed_episode_attempt_and_original_budget_without_later_scoring():
    def fail(owner, features, targets):
        raise RuntimeError("complete episode update fault")

    source, budget = learner(train=fail), session()
    result = run(source, budget=budget)
    assert result["status"] == "failed" and "complete episode update fault" in result["reason"]
    assert result["attempted_updates"] == 1 and result["attempted_training_examples"] == 3
    assert (
        result["completed_updates"]
        == result["completed_training_examples"]
        == budget.updates_completed
        == 0
    )
    assert [point["updates"] for point in result["checkpoints"]] == [0]
    assert result["score_forward_calls_completed"] == 1 and source.value[0] == 99


def test_should_keep_detached_episode_scope_when_opaque_checkpoint_copy_fails():
    source = learner()
    result = run(source, state=BrokenState())
    assert result["status"] == "failed" and "checkpoint copy failed" in result["reason"]
    assert (
        result["reserved_data_bytes"] == 216
        and result["episode_scope"]["training_transitions"] == 6
    )
    assert source.trace == [] and result["elapsed_seconds"] is None


def test_should_report_unknown_episode_scope_when_reserved_copy_is_incomplete(monkeypatch):
    original = probe_module._copy_batch
    calls = []

    def copy(item):
        calls.append(item)
        if len(calls) == 2:
            raise RuntimeError("second reserved packet copy failed")
        return original(item)

    monkeypatch.setattr(probe_module, "_copy_batch", copy)
    source = learner()
    result = run(source)
    assert result["status"] == "failed" and "second reserved packet copy failed" in result["reason"]
    assert result["reserved_data_bytes"] is None and result["episode_scope"] is None
    assert result["attempted_updates"] == result["score_forward_calls_attempted"] == 0
    assert source.trace == [] and result["elapsed_seconds"] is None
