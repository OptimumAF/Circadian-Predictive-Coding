"""Categorical packets cross the existing bounded stream without native learners."""

from dataclasses import replace
from math import log
from typing import Any

import numpy as np
import pytest

from src.app.coupled_attribution import AttributionBatch, AttributionStream
from src.app.stream_rehearsal import StreamRehearsal, StreamReplayPolicy
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget, ToyExecutionStopped
from src.core.categorical_probability_scores import probability_scores
from src.core.circadian_predictive_coding import ReplayRetentionBudget
from src.core.demonstration_episode import (
    DemonstrationEpisode,
    DemonstrationTransition,
    EpisodeReplaySchema,
    copy_episode,
    decode_episode,
    encode_episode,
    episode_arrays,
)
from src.core.shared_replay_schedule import ReplayAllocation
from test_trajectory_rehearsal import Learner as BinaryLearner


def packet(identity="packet-A", context="A", role="train", classes=3):
    offset = 0.15 if role == "final_test" else 0.0
    y = {"A": -0.75, "B": 0.0, "C": 0.75}[context]
    rotation = {"A": 0, "B": 1, "C": 2}[context]
    return DemonstrationEpisode(
        identity,
        context,
        role,
        tuple(
            DemonstrationTransition(
                step,
                (-0.75 + step * 0.5 + offset, y),
                (step + rotation) % classes,
                float(step == 2),
                (-0.75 + (step + 1) * 0.5 + offset, y),
                step == 2,
                classes,
            )
            for step in range(3)
        ),
        classes,
    )


def batch(identity, context="A", role="train", classes=3):
    episode = packet(identity, context, role, classes)
    features, targets = episode_arrays(episode)
    return AttributionBatch(
        role,
        context,
        tuple(f"{identity}:step:{index}" for index in range(3)),
        features,
        targets,
        episode,
        classes,
    )


def data():
    return (
        tuple(batch(f"train-{index}", context) for index, context in enumerate("ABCA")),
        tuple(batch(f"final-{context}", context, "final_test") for context in "ABC"),
    )


class Learner(BinaryLearner):
    def fork(self):
        self.control["forks"] += 1
        return Learner(self.control, self.state[0])

    def train_batch(self, features, targets):
        assert targets.dtype == np.float64 and targets.shape == (len(features), 1)
        assert np.isfinite(targets).all() and (targets == np.floor(targets)).all()
        assert (targets >= 0).all() and (targets < 3).all()
        return super().train_batch(features, targets)

    def predict(self, features):
        assert self.control["allow_final"] is True
        self.control["predictions"].append((len(self.control["trace"]), features.copy()))
        if self.control.get("bad_prediction"):
            return np.full((len(features), 1), 0.5)
        return np.tile(np.array([[0.2, 0.3, 0.5]]), (len(features), 1))


def setup(*, replay_limit=6, schema=None, **faults):
    clock = lambda: 0.0
    wake = ToyBudgetSession(ToyExecutionBudget(max_training_updates=4, max_wall_seconds=10), clock)
    replay = ToyBudgetSession(
        ToyExecutionBudget(max_replay_examples=replay_limit, max_wall_seconds=10), clock
    )
    source = Learner()
    source.control.update(replay=replay, predictions=[], allow_final=False, **faults)
    selected = StreamReplayPolicy(
        every_updates=2,
        rows_per_event=3,
        retention=ReplayRetentionBudget(6, 1936),
        allocation=ReplayAllocation(ReplayRetentionBudget(3, 968), ReplayRetentionBudget(3, 968)),
        seed=53,
    )
    helper = StreamRehearsal(
        source,
        wake,
        selected,
        replay_budget=replay,
        total_updates=4,
        input_dim=2,
        episode_schema=EpisodeReplaySchema(2, 3, 3) if schema is None else schema,
    )
    return source, wake, replay, helper


def stream(source, wake, helper, *, training=None, heldout=None, delay=1):
    defaults = data()
    return AttributionStream(
        source,
        defaults[0] if training is None else training,
        defaults[1] if heldout is None else heldout,
        checkpoint_updates=(0, 2, 4),
        label_delay=delay,
        budget=wake,
        rehearsal=helper,
    )


@pytest.mark.parametrize("classes", [3, 16])
def test_should_roundtrip_categorical_complete_packets_with_bound_header(classes):
    original = packet(classes=classes)
    schema = EpisodeReplaySchema(2, 3, classes)
    encoded, dummy = encode_episode(original, schema)
    assert encoded.shape == (1, 120) and schema.encoded_dim == 120
    assert encoded[0, 98] == classes
    assert encoded.nbytes + dummy.nbytes == 968
    assert decode_episode(encoded, dummy, schema) == original
    features, targets = episode_arrays(original)
    np.testing.assert_array_equal(targets[:, 0], [step.action for step in original.transitions])
    encoded.fill(0)
    features.fill(999)
    targets.fill(999)
    assert copy_episode(original) == original


def test_should_preserve_original_binary_wire_bytes_without_class_header():
    original = packet(classes=2)
    encoded, dummy = encode_episode(original, EpisodeReplaySchema(2, 3))

    def text(value, limit):
        raw = value.encode("utf-8")
        return [float(len(raw)), *map(float, raw), *([0.0] * (limit - len(raw)))]

    expected = text(original.episode_id, 64) + text(original.context, 32)
    for transition in original.transitions:
        expected.extend(transition.observation)
        expected.extend((float(transition.action), transition.reward))
        expected.extend(transition.next_observation)
        expected.append(float(transition.terminal))
    assert encoded.tobytes() == np.array([expected], dtype=np.float64).tobytes()
    assert encoded.shape == (1, 119)
    assert dummy.tobytes() == np.zeros((1, 1), dtype=np.float64).tobytes()


@pytest.mark.parametrize("kind", ["transition", "episode", "schema", "metric"])
@pytest.mark.parametrize("classes", [True, 1, 17, np.int64(3)])
def test_should_refuse_non_native_and_out_of_range_class_declarations(kind, classes):
    with pytest.raises(ValueError):
        if kind == "transition":
            replace(packet().transitions[0], num_classes=classes)
        elif kind == "episode":
            replace(packet(), num_classes=classes)
        elif kind == "schema":
            EpisodeReplaySchema(2, 3, classes)
        else:
            probability_scores(np.array([[0.2, 0.3, 0.5]]), np.array([[2.0]]), classes)


def test_should_refuse_other_class_schema_even_when_all_observed_actions_overlap():
    original = packet(classes=3)
    encoded, dummy = encode_episode(original, EpisodeReplaySchema(2, 3, 3))
    with pytest.raises(ValueError):
        decode_episode(encoded, dummy, EpisodeReplaySchema(2, 3, 4))
    with pytest.raises(ValueError):
        encode_episode(original, EpisodeReplaySchema(2, 3, 4))


@pytest.mark.parametrize("header", [2.0, 4.0, 3.5])
def test_should_refuse_replaced_or_non_integral_encoded_class_header(header):
    schema = EpisodeReplaySchema(2, 3, 3)
    encoded, dummy = encode_episode(packet(), schema)
    encoded[0, 98] = header
    with pytest.raises(ValueError):
        decode_episode(encoded, dummy, schema)


@pytest.mark.parametrize("classes", [3, 16])
def test_should_refuse_action_equal_to_declared_class_count(classes):
    with pytest.raises(ValueError):
        replace(packet(classes=classes).transitions[0], action=classes)


@pytest.mark.parametrize("component", ["episode", "transition"])
def test_should_revalidate_count_binding_when_copying_tampered_frozen_packet(component):
    original = packet()
    copied = copy_episode(original)
    changed = original if component == "episode" else original.transitions[0]
    object.__setattr__(changed, "num_classes", 4)
    with pytest.raises(ValueError):
        copy_episode(original)
    assert copied.num_classes == 3
    assert all(step.num_classes == 3 for step in copied.transitions)


@pytest.mark.parametrize(
    "predictions,targets,ce,accuracy",
    [
        (np.eye(3), np.array([[0.0], [1.0], [2.0]]), 0.0, 1.0),
        (np.array([[0.5, 0.5, 0.0]]), np.array([[1.0]]), log(2), 0.0),
        (np.array([[0.0, 1.0, 0.0]]), np.array([[0.0]]), -log(1e-12), 0.0),
    ],
)
def test_should_score_categorical_ce_clip_and_lowest_index_argmax_without_repair(
    predictions, targets, ce, accuracy
):
    before_predictions, before_targets = predictions.copy(), targets.copy()
    scores = probability_scores(predictions, targets, num_classes=3)
    assert scores["ce"] == pytest.approx(ce, abs=1e-12)
    assert scores["accuracy"] == accuracy
    np.testing.assert_array_equal(predictions, before_predictions)
    np.testing.assert_array_equal(targets, before_targets)


@pytest.mark.parametrize(
    "fault", ["dtype", "shape", "empty", "nan", "negative", "upper", "sum", "list"]
)
def test_should_refuse_invalid_categorical_probabilities_without_normalizing(fault):
    predictions: Any = np.array([[0.2, 0.3, 0.5]])
    targets = np.array([[2.0]])
    if fault == "dtype":
        predictions = predictions.astype(np.float32)
    elif fault == "shape":
        predictions = np.array([[0.5, 0.5]])
    elif fault == "empty":
        predictions, targets = np.zeros((0, 3)), np.zeros((0, 1))
    elif fault == "nan":
        predictions[0, 0] = np.nan
    elif fault == "negative":
        predictions[0] = [-0.1, 0.6, 0.5]
    elif fault == "upper":
        predictions[0] = [0, 0, 1.1]
    elif fault == "sum":
        predictions[0] = [0.2, 0.3, 0.4]
    else:
        predictions = predictions.tolist()
    with pytest.raises(ValueError):
        probability_scores(predictions, targets, num_classes=3)


@pytest.mark.parametrize("fault", ["fraction", "range", "nan", "dtype"])
def test_should_refuse_invalid_categorical_score_labels(fault):
    targets = np.array([[2.0]])
    if fault == "fraction":
        targets[0, 0] = 0.5
    elif fault == "range":
        targets[0, 0] = 3
    elif fault == "nan":
        targets[0, 0] = np.nan
    else:
        targets = targets.astype(np.float32)
    with pytest.raises(ValueError):
        probability_scores(np.array([[0.2, 0.3, 0.5]]), targets, num_classes=3)


@pytest.mark.parametrize("offset", [5e-7, 2e-6])
def test_should_use_fixed_absolute_probability_sum_tolerance_without_normalization(offset):
    predictions = np.array([[0.2, 0.3, 0.5 + offset]])
    if offset > 1e-6:
        with pytest.raises(ValueError):
            probability_scores(predictions, np.array([[2.0]]), 3)
    else:
        scores = probability_scores(predictions, np.array([[2.0]]), 3)
        assert scores["ce"] == pytest.approx(-log(0.5 + offset), abs=1e-15)
        assert predictions[0, 2] == 0.5 + offset


@pytest.mark.parametrize("delay", [0, 1])
def test_should_train_and_replay_whole_three_action_packets_before_final_only_scores(delay):
    source, wake, replay, helper = setup()
    owner = stream(source, wake, helper, delay=delay)
    owner.run_until(4)
    assert source.control["predictions"] == []
    source.control["allow_final"] = True
    result = owner.finish()
    assert source.state[0] == 0
    assert result["training_examples"] == 12 and result["completed_updates"] == 4
    assert replay.replay_examples_completed == 6
    assert [row[0] for row in source.control["trace"]] == [False, False, True, False, False, True]
    assert all(row[2].shape == (3, 2) and row[3].shape == (3, 1) for row in source.control["trace"])
    assert helper.report["episode_schema"] == dict(
        observation_dim=2, steps_per_episode=3, num_classes=3
    )
    assert helper.retention.example_count <= 6 and helper.retention.retained_bytes <= 1936
    assert len(source.control["predictions"]) == 9
    assert all(count == 6 for count, _ in source.control["predictions"])
    for event in helper.report["events"]:
        assert event["charged_exposures"] == 3
        assert (
            event["attempted_updates"]
            == event["returned_updates"]
            == event["committed_updates"]
            == 1
        )
        selected = event["selected_episodes"][0]
        assert selected["num_classes"] == 3
        assert [step["action"] for step in selected["transitions"]] in (
            [0, 1, 2],
            [1, 2, 0],
            [2, 0, 1],
        )
    for checkpoint in result["checkpoints"]:
        for score in checkpoint["scores"].values():
            assert score["ce"] == pytest.approx(-(log(0.2) + log(0.3) + log(0.5)) / 3)
            assert score["accuracy"] == pytest.approx(1 / 3)


def test_should_resume_same_categorical_runtime_under_original_authority_and_costs():
    reference_source, reference_wake, reference_replay, reference_helper = setup()
    source, wake, replay, helper = setup()
    reference = stream(reference_source, reference_wake, reference_helper)
    owner = stream(source, wake, helper)
    reference.run_until(4)
    reference_source.control["allow_final"] = True
    expected = reference.finish()
    owner.run_until(2)
    runtime, actor, original_budget = owner._runtime, owner._runtime.actor, wake.budget
    token = owner.pause()
    owner.resume(token)
    assert owner._runtime is runtime and runtime.actor is actor and wake.budget is original_budget
    assert replay.replay_examples_completed == 3
    assert source.control["predictions"] == []
    owner.run_until(4)
    source.control["allow_final"] = True
    assert owner.finish() == expected
    assert wake.updates_completed == reference_wake.updates_completed == 4
    assert replay.replay_examples_completed == reference_replay.replay_examples_completed == 6


@pytest.mark.parametrize("fault", ["tamper", "replace"])
def test_should_refuse_changed_original_class_schema_before_forks(fault):
    source, wake, _, helper = setup()
    if fault == "tamper":
        object.__setattr__(helper._episode_schema, "num_classes", 4)
    else:
        helper._episode_schema = EpisodeReplaySchema(2, 3, 4)
    with pytest.raises(ValueError):
        stream(source, wake, helper)
    assert source.control["forks"] == 0 and source.control["trace"] == []


@pytest.mark.parametrize("fault", ["batch_count", "episode_count", "final_count", "helper_count"])
def test_should_refuse_mixed_ingress_counts_before_binding_or_external_work(fault):
    source, wake, _, helper = setup(
        schema=EpisodeReplaySchema(2, 3, 4) if fault == "helper_count" else None
    )
    training, heldout = data()
    if fault == "batch_count":
        training = (replace(training[0], num_classes=4), *training[1:])
    elif fault == "episode_count":
        object.__setattr__(training[0].episode, "num_classes", 4)
    elif fault == "final_count":
        heldout = (batch("different-count", "A", "final_test", 4), *heldout[1:])
    with pytest.raises(ValueError):
        stream(source, wake, helper, training=training, heldout=heldout)
    assert source.control["forks"] == 0 and source.control["trace"] == []
    assert helper.report["bound"] is False


def test_should_preserve_complete_transition_charge_when_replay_training_fails():
    source, wake, replay, helper = setup(fail_replay=True)
    owner = stream(source, wake, helper)
    with pytest.raises(RuntimeError, match="whole episode replay fault"):
        owner.run_until(2)
    event = helper.report["events"][0]
    assert replay.replay_examples_completed == event["charged_exposures"] == 3
    assert event["attempted_updates"] == 1
    assert event["returned_updates"] == event["committed_updates"] == 0
    assert event["receipt"] is None and wake.updates_completed == 2
    assert source.state[0] == 0 and source.control["predictions"] == []


def test_should_refuse_partial_transition_budget_without_partial_episode_or_charge():
    source, wake, replay, helper = setup(replay_limit=2)
    owner = stream(source, wake, helper)
    owner.run_until(1)
    forks = source.control["forks"]
    with pytest.raises(ToyExecutionStopped):
        owner.run_until(2)
    assert replay.replay_examples_completed == 0
    assert source.control["forks"] == forks
    assert wake.updates_completed == len(source.control["trace"]) == 2
    assert helper.report["events"][0]["charged_exposures"] == 0


def test_should_refuse_wrong_output_shape_only_at_final_scoring_without_changing_wake_costs():
    source, wake, replay, helper = setup(bad_prediction=True)
    owner = stream(source, wake, helper)
    owner.run_until(4)
    before = owner.candidate_snapshot()
    source.control["allow_final"] = True
    with pytest.raises(ValueError):
        owner.finish()
    np.testing.assert_array_equal(owner._runtime.candidate_snapshot().state, before)
    assert wake.updates_completed == 4 and replay.replay_examples_completed == 6
    assert len(source.control["trace"]) == 6 and len(source.control["predictions"]) == 1
