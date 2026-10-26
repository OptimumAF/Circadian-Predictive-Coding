"""Isolated TRAIN fit controls validate all evidence and retain failed work.

Synthetic complete public histories use the existing environment-free frame
fixture. These checks perform small NumPy fits, with no native learner or game.
"""

from dataclasses import replace
import pickle
from typing import Any

import numpy as np
import pytest

from src.adapters import minigrid_linear_fit_control as control
from src.core.categorical_probability_margins import probability_margins
from src.core.demonstration_episode import DemonstrationEpisode, DemonstrationTransition
from test_minigrid_invariant_encoding import frame


def episode(identity="A-blue", *, actions=(0, 2, 6), cue=2, role="train"):
    initial = frame(wall_color=0, direction=0, cue_color=cue, key_color=cue)
    views = tuple(
        initial
        + frame(
            wall_color=index % 6,
            direction=index % 4,
            cue_color=cue,
            key_color=cue,
            door_state=index % 3,
        )
        for index in range(len(actions) + 1)
    )
    return DemonstrationEpisode(
        identity,
        "A",
        role,
        tuple(
            DemonstrationTransition(
                index, views[index], action, 0.0, views[index + 1], index == len(actions) - 1, 7
            )
            for index, action in enumerate(actions)
        ),
        7,
    )


def records():
    return (episode(), episode("A-red", actions=(1, 3), cue=0))


def test_should_report_original_episode_order_all_raw_probabilities_and_complete_costs():
    packets = records()
    before = pickle.dumps(packets)
    report = control.diagnose_train_linear_fit(packets)
    assert report["status"] == "complete", report["reason"]
    assert report["definition"] == "minigrid_categorical_linear_train_fit_v1"
    assert report["role"] == "train" and report["source_unchanged"] is True
    assert report["fit"]["feature_dim"] == 296 and report["fit"]["num_classes"] == 7
    assert report["fit"]["fit_rows"] == 5
    assert [row["episode_id"] for row in report["scores"]] == [p.episode_id for p in packets]
    for row, packet in zip(report["scores"], packets, strict=True):
        targets = np.array([[step.action] for step in packet.transitions], dtype=np.float64)
        probabilities = np.array(row["probabilities"], dtype=np.float64)
        actions = probabilities.argmax(axis=1)
        assert row["target_actions"] == targets[:, 0].astype(int).tolist()
        assert row["predicted_actions"] == actions.tolist()
        assert row["accuracy"] == pytest.approx(np.mean(actions == targets[:, 0]))
        assert row["exact_sequence_accuracy"] == float(np.all(actions == targets[:, 0]))
        ce = -np.log(
            np.clip(probabilities[np.arange(len(targets)), targets[:, 0].astype(int)], 1e-12, 1)
        ).mean()
        assert row["ce"] == pytest.approx(ce)
        assert row["action_margins"] == probability_margins(probabilities, targets, 7)
    work = report["work"]
    assert work["fit_calls_attempted"] == work["fit_calls_completed"] == 1
    assert work["fit_rows_attempted"] == work["fit_rows_completed"] == 5
    assert work["prediction_calls_attempted"] == work["prediction_calls_completed"] == 1
    assert work["prediction_rows_attempted"] == work["prediction_rows_completed"] == 5
    assert work["projection_rows_attempted"] == work["projection_rows_completed"] == 10
    assert work["projected_array_equivalent_bytes"] == 10 * 296 * 8
    assert work["feature_input_array_bytes"] == 5 * 296 * 8
    assert work["target_input_array_bytes"] == 5 * 8
    assert work["design_array_bytes"] == 5 * 297 * 8
    assert work["one_hot_array_bytes"] == 5 * 7 * 8
    assert work["coefficient_array_bytes"] == 297 * 7 * 8
    assert work["score_reductions_completed"] == 3 and work["score_rows_completed"] == 10
    assert work["native_forwards"] == work["optimizer_steps"] == work["benchmark_updates"] == 0
    for name in ("fit_seconds", "prediction_seconds", "elapsed_seconds"):
        assert work[name] >= 0
    assert pickle.dumps(packets) == before


@pytest.mark.parametrize(
    "fault",
    [
        "list",
        "empty",
        "too_many",
        "final_test",
        "probe_train",
        "probe_test",
        "extra",
        "last_extra",
        "incomplete",
        "classes",
        "width",
        "late_current",
        "late_next",
        "rows",
    ],
)
def test_should_refuse_whole_invalid_catalogue_before_fit_or_prediction(monkeypatch, fault):
    packets: Any = records()
    if fault == "list":
        packets = list(packets)
    elif fault == "empty":
        packets = ()
    elif fault == "too_many":
        packets = (packets[0],) * 17
    elif fault in ("final_test", "probe_train", "probe_test"):
        packets = (packets[0], replace(packets[1], role=fault))
    elif fault in ("extra", "last_extra"):
        object.__setattr__(
            packets[-1] if fault == "extra" else packets[-1].transitions[-1], "foreign", 1
        )
    elif fault == "incomplete":
        object.__setattr__(packets[-1].transitions[-1], "terminal", False)
    elif fault == "classes":
        object.__setattr__(packets[-1], "num_classes", 2)
    elif fault == "width":
        object.__setattr__(packets[-1].transitions[-1], "next_observation", frame())
    elif fault == "rows":
        packets = tuple(episode(str(index), actions=(2,) * 16) for index in range(9))
    else:
        last = packets[-1].transitions[-1]
        field = "observation" if fault == "late_current" else "next_observation"
        changed = list(getattr(last, field))
        changed[-2] += 0.01
        object.__setattr__(last, field, tuple(changed))
    monkeypatch.setattr(
        control, "fit_categorical_linear", lambda *a, **k: pytest.fail("bad source fit")
    )
    monkeypatch.setattr(
        control, "predict_categorical_linear", lambda *a, **k: pytest.fail("bad source predicted")
    )
    with pytest.raises(ValueError):
        control.diagnose_train_linear_fit(packets)


def test_should_fit_only_canonical_current_views_and_never_next_views_or_rewards(monkeypatch):
    packets = records()
    original_fit = control.fit_categorical_linear
    captured = []

    def fit(features, targets, count):
        captured.append((features.copy(), targets.copy(), count))
        return original_fit(features, targets, count)

    monkeypatch.setattr(control, "fit_categorical_linear", fit)
    first = control.diagnose_train_linear_fit(packets)
    last = packets[-1].transitions[-1]
    changed_next = frame(cue_color=4, key_color=4) * 2
    modified = (
        *packets[:-1],
        replace(
            packets[-1],
            transitions=(
                *packets[-1].transitions[:-1],
                replace(last, reward=1.0, next_observation=changed_next),
            ),
        ),
    )
    second = control.diagnose_train_linear_fit(modified)
    assert first["status"] == second["status"] == "complete"
    assert len(captured) == 2 and captured[0][2] == captured[1][2] == 7
    np.testing.assert_array_equal(captured[0][0], captured[1][0])
    np.testing.assert_array_equal(captured[0][1], captured[1][1])
    assert first["fit"] == second["fit"] and first["scores"] == second["scores"]


@pytest.mark.parametrize("phase", ["fit", "prediction"])
def test_should_preserve_attempted_and_completed_counts_on_solver_or_prediction_failure(
    monkeypatch, phase
):
    packets = records()
    before = pickle.dumps(packets)

    def fail(*args, **kwargs):
        raise np.linalg.LinAlgError("injected independent solver failure")

    monkeypatch.setattr(
        control, "fit_categorical_linear" if phase == "fit" else "predict_categorical_linear", fail
    )
    report = control.diagnose_train_linear_fit(packets)
    assert (
        report["status"] == "failed" and "injected independent solver failure" in report["reason"]
    )
    assert report["source_unchanged"] is True and not report["scores"]
    work = report["work"]
    assert work["fit_calls_attempted"] == 1 and work["fit_rows_attempted"] == 5
    assert work["fit_calls_completed"] == (phase == "prediction")
    assert work["prediction_calls_attempted"] == (phase == "prediction")
    assert work["prediction_calls_completed"] == work["prediction_rows_completed"] == 0
    assert work["failure_work_may_be_unobserved"] is True
    assert pickle.dumps(packets) == before


@pytest.mark.parametrize("phase", ["fit", "prediction"])
def test_should_refuse_foreign_source_change_without_repair_and_keep_paid_costs(monkeypatch, phase):
    packets = records()
    original = getattr(
        control, "fit_categorical_linear" if phase == "fit" else "predict_categorical_linear"
    )

    def corrupt(*args, **kwargs):
        result = original(*args, **kwargs)
        object.__setattr__(packets[-1], "role", "final_test")
        return result

    monkeypatch.setattr(
        control,
        "fit_categorical_linear" if phase == "fit" else "predict_categorical_linear",
        corrupt,
    )
    report = control.diagnose_train_linear_fit(packets)
    assert report["status"] == "failed" and report["source_unchanged"] is False
    assert packets[-1].role == "final_test" and not report["scores"]
    assert report["work"]["fit_calls_completed"] == 1
    assert report["work"]["fit_rows_completed"] == 5
    if phase == "fit":
        assert report["work"]["prediction_calls_attempted"] == 0
    else:
        assert report["work"]["prediction_calls_completed"] == 1
