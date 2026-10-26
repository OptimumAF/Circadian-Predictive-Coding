"""The fixed A-only diagnostic admits exactly its two original TRAIN cues."""

from dataclasses import replace
import pickle
from typing import Any

import numpy as np
import pytest

from src.adapters import minigrid_policy_evaluation as evaluation
from src.core.demonstration_episode import episode_arrays
from src.core.minigrid_key_door import predefined_layouts
from src.core.minigrid_public_history import encode_public_history_episode
from test_minigrid_history_evaluation import checkpoint, history_case as history_case
from test_minigrid_learning_pilot import evaluator as evaluator


def evaluate_a(case, *, method="bp", history=True):
    state = checkpoint(case, method, width=None if history else 148)
    state.completed_updates, state.training_examples = 32, 288
    return evaluation.evaluate_checkpoint(
        case.source,
        state,
        method=method,
        updates=32,
        layouts=predefined_layouts()[:1],
        role="train",
        sources=case.sources[:2],
        budget=case.budget,
        check_boundary=case.guard,
        **({"public_history": True} if history else {}),
    )


@pytest.mark.parametrize("method", ["bp", "pc", "neutral_cpc", "gated_cpc", "recurrent_bc"])
def test_should_evaluate_only_both_a_cues_with_owned_inputs_and_original_budget(
    history_case, method
):
    case = history_case
    checkpoint(case, method)
    before = pickle.dumps((case.sources, case.source.snapshot_state()))
    budget = (case.budget.budget, case.budget.started_at, case.budget.updates_completed)
    report = evaluate_a(case, method=method)
    assert report["status"] == "complete" and report["state_unchanged"] is True
    assert report["updates"] == 32 and report["role"] == "train"
    assert pickle.dumps((case.sources, case.source.snapshot_state())) == before
    assert (case.budget.budget, case.budget.started_at, case.budget.updates_completed) == budget
    assert [(row["context"], row["cue"]) for row in report["scores"]] == [
        ("A", "blue"),
        ("A", "red"),
    ]
    assert all(row["ce"] == pytest.approx(float(np.log(2))) for row in report["scores"])
    assert all(row["accuracy"] == row["exact_sequence_accuracy"] == 0 for row in report["scores"])
    assert len(case.work["predicts"]) == 6 and not case.work.get("trains")
    width = 148 if method == "recurrent_bc" else 296
    for index, packet in enumerate(case.sources[:2]):
        actual = case.work["predicts"][index][1]
        expected = (
            packet.features
            if width == 148
            else episode_arrays(encode_public_history_episode(packet.episode))[0]
        )
        np.testing.assert_array_equal(actual, expected)
        assert actual.shape == (9, width) and not np.shares_memory(actual, packet.features)
    assert [(game.layout.context, game.cue, game.role) for game in case.work["environments"]] == [
        ("A", "blue", "train"),
        ("A", "red", "train"),
    ]
    for index, (game, slot) in enumerate(
        zip(case.work["environments"], report["episodes"], strict=True)
    ):
        assert game.actions == [0, 0] and game.closed
        assert slot["success"] is True and slot["return"] == 0.5
        initial = game.transitions[0]["observation"]
        for step, decision in enumerate(slot["predictions"]):
            current = game.transitions[step]["observation"]
            expected = current if width == 148 else initial + current
            np.testing.assert_array_equal(
                case.work["predicts"][2 + index * 2 + step][1], [expected]
            )
            if width == 296:
                assert tuple(decision["features"]) == expected
            else:
                assert "features" not in decision
            assert tuple(decision["observation"]) == current and decision["action"] == 0
    assert (
        case.work["environments"][0].transitions[0]["observation"]
        != case.work["environments"][1].transitions[0]["observation"]
    )
    work = report["work"]
    assert work["raw_score_forwards_completed"] == 2 and work["raw_score_rows_completed"] == 18
    assert work["live_prediction_calls_returned"] == 4
    assert work["environment_resets_completed"] == 2 and work["environment_steps_returned"] == 4
    assert work["live_sessions_started"] == (2 if width == 148 else 0)
    assert len(case.work.get("sessions", [])) == (2 if width == 148 else 0)
    assert work["history_raw_episodes_encoded"] == (0 if width == 148 else 2)
    assert work["history_live_feature_rows"] == (0 if width == 148 else 4)


def test_should_allow_same_fixed_a_pair_with_original_default_148_encoding(history_case):
    report = evaluate_a(history_case, history=False)
    assert report["status"] == "complete" and "public_history" not in report
    assert len(report["scores"]) == len(report["episodes"]) == 2
    assert all(values.shape[1] == 148 for _, values in history_case.work["predicts"])
    assert not any(key.startswith("history_") for key in report["work"])


@pytest.mark.parametrize(
    "fault",
    [
        "context",
        "order",
        "missing",
        "duplicate",
        "encoded",
        "role",
        "feature_mismatch",
        "updates_bool",
        "geometry_rotation",
        "geometry_branch",
        "geometry_swap",
        "geometry_color",
        "extra_source",
    ],
)
def test_should_refuse_nonfixed_a_source_or_layout_before_fork_and_environment(history_case, fault):
    case = history_case
    state = checkpoint(case)
    state.completed_updates, state.training_examples = 32, 288
    layouts, sources, updates = predefined_layouts()[:1], case.sources[:2], 32
    if fault == "context":
        layouts = predefined_layouts()[1:2]
        sources = case.sources[2:4]
    elif fault == "order":
        sources = tuple(reversed(sources))
    elif fault == "missing":
        sources = sources[:1]
    elif fault == "duplicate":
        sources = (sources[0], sources[0])
    elif fault == "encoded":
        episode = encode_public_history_episode(sources[0].episode)
        values, targets = episode_arrays(episode)
        sources = (
            replace(sources[0], episode=episode, features=values, targets=targets),
            sources[1],
        )
    elif fault == "role":
        first = replace(
            sources[0], role="probe_train", episode=replace(sources[0].episode, role="probe_train")
        )
        sources = (first, sources[1])
    elif fault == "feature_mismatch":
        sources[0].features[0, 0] = 0.125
    elif fault == "updates_bool":
        updates = True
    elif fault.startswith("geometry_"):
        changes: dict[str, dict[str, Any]] = {
            "geometry_rotation": {"rotation": 1},
            "geometry_branch": {"branch_row": 4},
            "geometry_swap": {"swap_keys": True},
            "geometry_color": {"wall_color": "red"},
        }
        change = changes[fault]
        layouts = (replace(layouts[0], **change),)
    else:
        sources = case.sources
    with pytest.raises(ValueError):
        evaluation.evaluate_checkpoint(
            case.source,
            state,
            method="bp",
            updates=updates,
            layouts=layouts,
            role="train",
            sources=sources,
            budget=case.budget,
            check_boundary=case.guard,
            public_history=True,
        )
    assert not case.work.get("forks") and not case.work.get("predicts")
    assert not case.work.get("environments") and case.budget.updates_completed == 0


@pytest.mark.parametrize("fault", ["early_endpoint", "a_layout", "teacher_labels"])
def test_should_preserve_endpoint_d_empty_label_gate_with_additional_a_admission(
    history_case, fault
):
    case = history_case
    state = checkpoint(case)
    updates, layouts, sources = 128, predefined_layouts()[-1:], ()
    if fault == "early_endpoint":
        updates = 32
    elif fault == "a_layout":
        layouts = predefined_layouts()[:1]
    else:
        sources = case.sources[:2]
    with pytest.raises(ValueError):
        evaluation.evaluate_checkpoint(
            case.source,
            state,
            method="bp",
            updates=updates,
            layouts=layouts,
            role="final_test",
            sources=sources,
            budget=case.budget,
            check_boundary=case.guard,
            public_history=True,
        )
    assert not case.work.get("forks") and not case.work.get("environments")
