"""Causal public first/current frames retain complete episode and probe boundaries."""

from dataclasses import replace
import pickle
from typing import Any

import numpy as np
import pytest

from src.adapters.torch_feature_learner import TorchFeatureLearner, TorchFeaturePolicy
from src.adapters.torch_feature_rehearsal_pilot import _same_state
from src.adapters.torch_recurrent_learner import TorchRecurrentPolicy
from src.app.coupled_attribution import AttributionBatch, _validate_batch
from src.app.fresh_task_probe import run_fresh_task_probe
from src.app.toy_execution_budget import ToyBudgetSession
from src.core.demonstration_episode import (
    DemonstrationEpisode,
    DemonstrationTransition,
    EpisodeReplaySchema,
    decode_episode,
    encode_episode,
    episode_arrays,
)
from src.core.minigrid_public_history import (
    PublicInitialFrameHistory,
    encode_public_history_episode,
)
from src.shared.torch_runtime import require_torch
from test_fresh_task_probe import session
from test_minigrid_probe_admission import plain, source


def frame(step=0):
    values: list[float] = []
    for pixel in range(49):
        values.extend(
            (
                2 * ((pixel + step) % 11) / 10 - 1,
                2 * ((pixel + step) % 6) / 5 - 1,
                float((pixel + step) % 3 - 1),
            )
        )
    return (*values, float(2 * (step % 4) / 3 - 1))


def episode(*, role="train", context="A", identity="history-A", steps=3):
    return DemonstrationEpisode(
        identity,
        context,
        role,
        tuple(
            DemonstrationTransition(
                index,
                frame(index),
                index % 7,
                0.5 if index == steps - 1 else -0.25,
                frame(index + 1),
                index == steps - 1,
                7,
            )
            for index in range(steps)
        ),
        7,
    )


def batch(record):
    features, targets = episode_arrays(record)
    return AttributionBatch(
        record.role,
        record.context,
        tuple(f"{record.episode_id}:step:{row.step}" for row in record.transitions),
        features,
        targets,
        record,
        7,
    )


def test_should_encode_only_initial_and_current_frame_and_reset_between_episodes():
    history = PublicInitialFrameHistory()
    with pytest.raises(ValueError):
        history.encode(frame())
    initial, current, later = frame(), frame(1), frame(2)
    assert history.reset(initial) == initial + initial
    assert history.encode(current) == initial + current
    assert history.encode(later) == initial + later
    assert history.reset(current) == current + current
    assert history.encode(later) == current + later
    history.clear()
    with pytest.raises(ValueError):
        history.encode(current)


@pytest.mark.parametrize(
    "fault", ["list", "short", "double", "nan", "inf", "range", "bool", "numpy"]
)
def test_should_clear_old_initial_frame_before_refusing_invalid_new_episode_reset(fault):
    history = PublicInitialFrameHistory()
    history.reset(frame())
    invalid: Any = list(frame(1))
    if fault == "short":
        invalid = tuple(invalid[:-1])
    elif fault == "double":
        invalid = tuple(invalid + invalid)
    elif fault != "list":
        invalid[0] = {
            "nan": float("nan"),
            "inf": float("inf"),
            "range": 1.1,
            "bool": True,
            "numpy": np.float64(0.0),
        }[fault]
        invalid = tuple(invalid)
    with pytest.raises(ValueError):
        history.reset(invalid)
    with pytest.raises(ValueError):
        history.encode(frame(2))
    assert history.reset(frame(3)) == frame(3) + frame(3)


def test_should_normalize_native_integer_coordinates_but_refuse_invalid_current_without_losing_initial():
    history = PublicInitialFrameHistory()
    assert history.reset((0,) * 148) == (0.0,) * 296
    invalid = (float("nan"),) + frame(1)[1:]
    with pytest.raises(ValueError):
        history.encode(invalid)
    assert history.encode(frame(1)) == (0.0,) * 148 + frame(1)
    assert all(type(value) is float for value in history.encode(frame(1)))


@pytest.mark.parametrize("role", ["train", "final_test", "probe_train", "probe_test"])
def test_should_transform_owned_complete_episode_without_changing_metadata_actions_or_outcomes(
    role,
):
    original = episode(role=role)
    before = pickle.dumps(original)
    encoded = encode_public_history_episode(original)
    assert encoded is not original and encoded.transitions is not original.transitions
    assert (encoded.episode_id, encoded.context, encoded.role, encoded.num_classes) == (
        original.episode_id,
        original.context,
        role,
        7,
    )
    initial = original.transitions[0].observation
    for old, new in zip(original.transitions, encoded.transitions, strict=True):
        assert new is not old
        assert new.observation == initial + old.observation
        assert new.next_observation == initial + old.next_observation
        assert (new.step, new.action, new.reward, new.terminal, new.num_classes) == (
            old.step,
            old.action,
            old.reward,
            old.terminal,
            old.num_classes,
        )
    assert pickle.dumps(original) == before
    features, targets = episode_arrays(encoded)
    assert features.shape == (3, 296) and targets.shape == (3, 1)
    _validate_batch(batch(encoded), role, set())
    object.__setattr__(original.transitions[0], "reward", 0.75)
    assert encoded.transitions[0].reward == -0.25


def test_should_not_change_any_current_feature_when_only_future_terminal_evidence_changes():
    original = episode()
    later = replace(
        original,
        transitions=(
            *original.transitions[:-1],
            replace(original.transitions[-1], action=6, reward=-1.0, next_observation=frame(13)),
        ),
    )
    before = encode_public_history_episode(original)
    after = encode_public_history_episode(later)
    assert [row.observation for row in before.transitions] == [
        row.observation for row in after.transitions
    ]
    assert [row.next_observation for row in before.transitions[:-1]] == [
        row.next_observation for row in after.transitions[:-1]
    ]
    assert before.transitions[-1].next_observation != after.transitions[-1].next_observation
    assert before.transitions[0].observation == frame() + frame()


def test_should_own_extracted_arrays_and_roundtrip_all_train_transition_fields_through_replay():
    encoded = encode_public_history_episode(episode())
    schema = EpisodeReplaySchema(296, 3, 7)
    wire, dummy = encode_episode(encoded, schema)
    restored = decode_episode(wire, dummy, schema)
    assert schema.encoded_dim == 1884 and wire.nbytes + dummy.nbytes == 15080
    assert restored == encoded and restored is not encoded
    first, labels = episode_arrays(encoded)
    second, other_labels = episode_arrays(encoded)
    assert not np.shares_memory(first, second) and not np.shares_memory(labels, other_labels)
    first.fill(0)
    labels.fill(6)
    wire.fill(0)
    assert restored == encoded
    np.testing.assert_array_equal(second[0], frame() + frame())
    np.testing.assert_array_equal(other_labels[:, 0], [0.0, 1.0, 2.0])


@pytest.mark.parametrize("role", ["final_test", "probe_train", "probe_test"])
def test_should_keep_history_episode_roles_out_of_train_replay(role):
    encoded = encode_public_history_episode(episode(role=role))
    with pytest.raises(ValueError, match="training"):
        encode_episode(encoded, EpisodeReplaySchema(296, 3, 7))


@pytest.mark.parametrize(
    "fault", ["double_encoding", "wrong_width", "discontinuous", "nonfinite", "not_episode"]
)
def test_should_refuse_malformed_or_previously_encoded_input_episode(fault):
    record: Any = episode()
    if fault == "double_encoding":
        record = encode_public_history_episode(record)
    elif fault == "wrong_width":
        record = DemonstrationEpisode(
            "small", "A", "train", (DemonstrationTransition(0, (0.0,), 0, 0.5, (1.0,), True, 7),), 7
        )
    elif fault == "discontinuous":
        object.__setattr__(record.transitions[1], "observation", frame(12))
    elif fault == "nonfinite":
        object.__setattr__(record.transitions[0], "observation", (float("nan"),) + frame()[1:])
    else:
        record = object()
    with pytest.raises(ValueError):
        encode_public_history_episode(record)


def test_should_admit_sixteen_separate_history_probe_episodes_with_original_entry_cap():
    training = tuple(
        batch(
            encode_public_history_episode(
                episode(
                    role="probe_train",
                    context=f"E-{'blue' if index % 2 == 0 else 'red'}",
                    identity=f"history-train-{index}",
                    steps=9,
                )
            )
        )
        for index in range(16)
    )
    evaluation = tuple(
        batch(
            encode_public_history_episode(
                episode(
                    role="probe_test", context=f"E-{cue}", identity=f"history-test-{cue}", steps=9
                )
            )
        )
        for cue in ("blue", "red")
    )
    learner, budget = source(), session(updates=16)
    report = run_fresh_task_probe(
        learner,
        np.array([0.0]),
        training,
        evaluation,
        budget=budget,
        checkpoint_updates=(0, 4, 8, 16),
        timer=lambda: 0.0,
        decision_diagnostics=True,
    )
    assert report["status"] == "complete", report["reason"]
    assert report["reserved_data_bytes"] == 384912
    assert report["completed_updates"] == budget.updates_completed == 16
    assert (
        report["completed_training_examples"] == 144
        and report["score_forward_calls_completed"] == 8
    )
    assert all(event[2].shape == (9, 296) for event in learner.trace if event[0] == "train")


@pytest.mark.parametrize("width", [295, 297])
def test_should_refuse_neighboring_history_probe_widths_before_external_calls(width):
    training = (plain("train", "probe_train", width=width),)
    evaluation = (plain("test", "probe_test", width=width),)
    learner = source()
    with pytest.raises(ValueError):
        run_fresh_task_probe(
            learner,
            np.array([0.0]),
            training,
            evaluation,
            budget=session(updates=1),
            checkpoint_updates=(0, 1),
            timer=lambda: 0.0,
        )
    assert learner.trace == []


def test_should_keep_65536_entry_probe_limit_for_wider_features_before_budget_or_fork(monkeypatch):
    training = (plain("train", "probe_train", width=296, rows=128),)
    evaluation = (plain("test", "probe_test", width=296, rows=128),)
    learner = source()

    def forbidden(*args, **kwargs):
        pytest.fail("oversized history input reached an external budget callback")

    monkeypatch.setattr(ToyBudgetSession, "before_final", forbidden)
    with pytest.raises(ValueError, match="65536"):
        run_fresh_task_probe(
            learner,
            np.array([0.0]),
            training,
            evaluation,
            budget=session(updates=1),
            checkpoint_updates=(0, 1),
            timer=lambda: 0.0,
        )
    assert learner.trace == []


@pytest.mark.parametrize("method", ["bp", "pc", "neutral_cpc", "gated_cpc"])
def test_should_admit_exact296_history_policies_and_preserve_original_dimensions(
    method,
):
    policy = TorchFeaturePolicy(method, 296, 8, 17, num_classes=7)
    assert (policy.feature_dim, policy.hidden_dim, policy.num_classes, policy.seed) == (
        296,
        8,
        7,
        17,
    )
    assert policy.learning_rate == 0.03 and policy.momentum == 0.9
    assert policy.inference_steps == 2 and policy.inference_learning_rate == 0.04
    assert replace(policy, feature_dim=148).feature_dim == 148
    for width in (295, 297):
        with pytest.raises(ValueError):
            replace(policy, feature_dim=width)
    assert TorchRecurrentPolicy(148, 8, 7, 17).feature_dim == 148
    assert TorchRecurrentPolicy(296, 8, 7, 17).feature_dim == 296


@pytest.mark.parametrize("method", ["bp", "pc", "neutral_cpc", "gated_cpc"])
def test_should_train_one_native_history_episode_on_owned_fork_and_restore_full_state(method):
    # Frozen qualification allowance: four methods, one9-row update and one
    # fork readout each. No environment or comparative efficacy assertion.
    torch = require_torch()
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    source_learner = TorchFeatureLearner(TorchFeaturePolicy(method, 296, 8, 17, num_classes=7))
    original = source_learner.snapshot_state()
    disposable = source_learner.fork()
    assert disposable is not source_learner
    assert _same_state(original, disposable.snapshot_state())
    features, targets = episode_arrays(encode_public_history_episode(episode(steps=9)))
    disposable.train_batch(features, targets)
    trained = disposable.snapshot_state()
    assert trained.completed_updates == 1 and trained.training_examples == 9
    assert _same_state(original, source_learner.snapshot_state())
    readout = disposable.fork()
    assert _same_state(trained, readout.snapshot_state())
    prediction = readout.predict(features)
    assert prediction.shape == (9, 7) and prediction.dtype == np.float64
    assert np.isfinite(prediction).all()
    np.testing.assert_allclose(prediction.sum(axis=1), 1.0, rtol=0, atol=1e-6)
    assert _same_state(trained, readout.snapshot_state())
    readout.restore_state(original)
    assert _same_state(original, readout.snapshot_state())
    assert _same_state(trained, disposable.snapshot_state())
    with torch.no_grad():
        trained.head_state["weight_feature_hidden"].add_(1.0)
    assert not _same_state(trained, disposable.snapshot_state())
    assert _same_state(original, source_learner.snapshot_state())
