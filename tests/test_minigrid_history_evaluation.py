"""History-aware frozen readouts use each episode's own public first frame.

All environments and learners here are owned fakes; no optional runtime, SGD,
native physics or final-test evidence is needed to verify the boundary.
"""

from dataclasses import replace
import pickle
from typing import Any

import numpy as np
import pytest

from src.adapters import minigrid_policy_evaluation as evaluation
from src.adapters.torch_feature_learner import TorchFeaturePolicy
from src.adapters.torch_recurrent_learner import TorchRecurrentPolicy
from src.core.minigrid_key_door import predefined_layouts
from src.core.minigrid_public_history import encode_public_history_episode
from src.core.demonstration_episode import episode_arrays
from test_minigrid_learning_pilot import Environment, evaluator as evaluator


class CueFrameEnvironment(Environment):
    def __init__(self, layout, cue, *args, **kwargs):
        super().__init__(layout, cue, *args, **kwargs)
        offset = {"A": 0.0, "B": 0.125, "C_train": 0.25, "D": 0.375}[layout.context]
        self.observation = (offset + (0.25 if cue == "blue" else 0.5),) + self.observation[1:]


@pytest.fixture
def history_case(monkeypatch, evaluator):
    monkeypatch.setattr(evaluation, "MiniGridKeyDoor", CueFrameEnvironment)
    return evaluator


def checkpoint(case, method="bp", width=None):
    policy: Any
    if method == "recurrent_bc":
        policy = TorchRecurrentPolicy(148 if width is None else width, 8, 7, 17)
    else:
        policy = TorchFeaturePolicy(method, 296 if width is None else width, 8, 17, num_classes=7)
    case.source.policy = policy
    case.source.state.policy = policy
    state = case.source.snapshot_state()
    state.completed_updates, state.training_examples = 128, 1152
    return state


def readout(case, *, method="bp", final=False, state=None, **options):
    if state is None:
        state = checkpoint(case, method)
    layouts = predefined_layouts()
    return evaluation.evaluate_checkpoint(
        case.source,
        state,
        method=method,
        updates=128,
        layouts=(layouts[-1],) if final else layouts[:3],
        role="final_test" if final else "train",
        sources=() if final else case.sources,
        budget=case.budget,
        check_boundary=case.guard,
        **options,
    )


@pytest.mark.parametrize("method", ["bp", "pc", "neutral_cpc", "gated_cpc"])
def test_should_score_original_train_sources_with_owned_history_and_reset_each_live_episode(
    history_case, method
):
    case = history_case
    state = checkpoint(case, method)
    before = pickle.dumps((case.sources, case.source.snapshot_state(), state))
    budget_binding = (case.budget.budget, case.budget.started_at, case.budget.updates_completed)
    report = readout(case, method=method, state=state, public_history=True)
    assert report["status"] == "complete" and report["state_unchanged"] is True
    assert pickle.dumps((case.sources, case.source.snapshot_state(), state)) == before
    assert (
        case.budget.budget,
        case.budget.started_at,
        case.budget.updates_completed,
    ) == budget_binding
    assert len(case.work["predicts"]) == 18 and not case.work.get("trains")
    assert not case.work.get("sessions")
    for index, packet in enumerate(case.sources):
        _, actual = case.work["predicts"][index]
        expected, _ = episode_arrays(encode_public_history_episode(packet.episode))
        np.testing.assert_array_equal(actual, expected)
        assert actual.shape == (9, 296) and packet.features.shape == (9, 148)
        assert not np.shares_memory(actual, packet.features)
    assert [(score["context"], score["cue"]) for score in report["scores"]] == [
        (context, cue) for context in ("A", "B", "C_train") for cue in ("blue", "red")
    ]
    assert all(score["ce"] == pytest.approx(float(np.log(2))) for score in report["scores"])
    assert all(
        score["accuracy"] == score["exact_sequence_accuracy"] == 0 for score in report["scores"]
    )
    for index, (game, slot) in enumerate(
        zip(case.work["environments"], report["episodes"], strict=True)
    ):
        initial = game.transitions[0]["observation"]
        assert game.closed and game.actions == [0, 0]
        assert slot["source_role"] == "train" and slot["success"] is True
        assert slot["return"] == 0.5 and slot["state_unchanged"] is True
        for step, decision in enumerate(slot["predictions"]):
            observation = game.transitions[step]["observation"]
            expected = initial + observation
            assert tuple(decision["observation"]) == observation
            assert tuple(decision["features"]) == expected
            np.testing.assert_array_equal(
                case.work["predicts"][6 + index * 2 + step][1], [expected]
            )
            assert decision["probabilities"] == [0.5, 0.5, 0, 0, 0, 0, 0]
            assert decision["action"] == 0
            assert not any(key in decision for key in ("target", "cue", "context", "reward"))
        assert slot["predictions"][0]["features"][:148] == slot["predictions"][0]["features"][148:]
        assert tuple(slot["predictions"][1]["features"][:148]) == initial
        assert initial[0] != slot["predictions"][1]["observation"][0]
    work = report["work"]
    assert work["raw_score_forwards_completed"] == 6 and work["raw_score_rows_completed"] == 54
    assert work["live_prediction_calls_returned"] == 12 and work["live_sessions_started"] == 0
    assert work["environment_resets_completed"] == 6 and work["environment_steps_returned"] == 12
    assert work["history_episodes_started"] == work["history_raw_episodes_encoded"] == 6
    assert work["history_live_feature_rows"] == 12
    assert work["history_live_feature_array_bytes"] == 12 * 296 * 8
    assert work["history_raw_feature_array_bytes"] == 6 * 9 * 296 * 8
    assert work["history_raw_target_array_bytes"] == 6 * 9 * 8
    assert report["public_history"]["prediction_feature_dim"] == 296
    assert report["public_history"]["initial_numeric_bytes"] == 1184


def test_should_keep_recurrent_148_sessions_and_raw_default_encoding_with_history_flag(
    history_case,
):
    case = history_case
    report = readout(case, method="recurrent_bc", public_history=True)
    assert report["status"] == "complete"
    assert (
        len(case.work["sessions"]) == 6 and len({id(value) for value in case.work["sessions"]}) == 6
    )
    assert all(session.completed_steps == 2 for session in case.work["sessions"])
    assert all(values.shape == (9, 148) for _, values in case.work["predicts"][:6])
    assert all(values.shape == (1, 148) for _, values in case.work["predicts"][6:])
    assert all(
        "features" not in decision
        for slot in report["episodes"]
        for decision in slot["predictions"]
    )
    assert report["work"]["live_sessions_started"] == 6
    assert report["public_history"]["applied"] is False
    assert report["public_history"]["prediction_feature_dim"] == 148
    assert report["public_history"]["initial_numeric_bytes"] == 0
    assert all(value == 0 for key, value in report["work"].items() if key.startswith("history_"))
    assert case.budget.updates_completed == 0 and not case.work.get("trains")


def test_should_preserve_148_default_report_and_explicit_false_exactly(history_case):
    case = history_case
    state = checkpoint(case, width=148)
    first = readout(case, state=state)
    second = readout(case, state=state, public_history=False)
    assert first == second
    assert "public_history" not in first
    assert not any(key.startswith("history_") for key in first["work"])
    assert all(values.shape[1] == 148 for _, values in case.work["predicts"])
    assert all(
        "features" not in decision for slot in first["episodes"] for decision in slot["predictions"]
    )


def test_should_support_target_free_fake_d_history_without_constructing_teacher_packets(
    history_case,
):
    case = history_case
    case.sources = ()
    report = readout(case, final=True, public_history=True)
    assert report["status"] == "complete" and report["scores"] == []
    assert report["work"]["raw_score_forwards_attempted"] == 0
    assert [(game.layout.context, game.role) for game in case.work["environments"]] == [
        ("D", "final_test"),
        ("D", "final_test"),
    ]
    assert len(case.work["predicts"]) == 4 and not case.work.get("trains")
    assert all(values.shape == (1, 296) for _, values in case.work["predicts"])


@pytest.mark.parametrize("invalid", [None, 0, 1, "true", np.bool_(True)])
def test_should_refuse_non_boolean_history_before_fork_or_environment(history_case, invalid):
    with pytest.raises(ValueError):
        readout(history_case, public_history=invalid)
    assert not history_case.work.get("forks") and not history_case.work.get("environments")


@pytest.mark.parametrize("history,width", [(False, 296), (True, 148)])
def test_should_bind_feedforward_checkpoint_width_to_explicit_encoding_before_fork(
    history_case, history, width
):
    state = checkpoint(history_case, width=width)
    with pytest.raises(evaluation.MiniGridEvaluationFailed):
        readout(history_case, state=state, public_history=history)
    assert not history_case.work.get("forks") and not history_case.work.get("predicts")


@pytest.mark.parametrize("fault", ["already_encoded", "role", "input_mismatch"])
def test_should_keep_original_148_train_source_admission_before_history_conversion(
    history_case, fault
):
    case = history_case
    first = case.sources[0]
    if fault == "already_encoded":
        episode = encode_public_history_episode(first.episode)
        values, targets = episode_arrays(episode)
        first = replace(first, episode=episode, features=values, targets=targets)
    elif fault == "role":
        first = replace(
            first, role="probe_train", episode=replace(first.episode, role="probe_train")
        )
    else:
        first.features[0, 0] = 0.125
    case.sources = (first, *case.sources[1:])
    with pytest.raises(ValueError):
        readout(case, public_history=True)
    assert not case.work.get("forks") and not case.work.get("predicts")


@pytest.mark.parametrize("target", ["source", "scorer", "optimizer", "checkpoint"])
def test_should_refuse_changed_checkpoint_or_source_after_paid_raw_history_score(
    history_case, target
):
    case = history_case
    state = checkpoint(case)

    def mutate(owner):
        if target == "checkpoint":
            state.parameters["weight"] += 1
            return
        selected = case.source if target == "source" else owner
        if target == "optimizer":
            selected.state.optimizer_state["momentum"] += 1
        else:
            selected.state.parameters["weight"] += 1

    case.work["on_predict"] = mutate
    with pytest.raises(evaluation.MiniGridEvaluationFailed) as raised:
        readout(case, state=state, public_history=True)
    report = raised.value.report
    assert report["work"]["raw_score_forwards_completed"] == 1
    assert report["work"]["raw_score_rows_completed"] == 9
    assert not case.work.get("environments") and len(case.work["predicts"]) == 1
    assert case.budget.updates_completed == 0


@pytest.mark.parametrize("fault", ["prediction", "probability", "step", "close_after_step"])
def test_should_retain_history_decision_and_paid_prefix_when_fake_live_work_fails(
    history_case, fault
):
    case = history_case
    if fault == "close_after_step":
        case.work.update(step_error=True, close_error=True)
    else:
        case.work[
            {
                "prediction": "prediction_error",
                "probability": "bad_probability",
                "step": "step_error",
            }[fault]
        ] = True
    with pytest.raises(evaluation.MiniGridEvaluationFailed) as raised:
        readout(case, final=True, public_history=True)
    report = raised.value.report
    assert len(report["episodes"]) == 1 and len(case.work["environments"]) == 1
    slot = report["episodes"][0]
    assert slot["status"] == "failed" and case.work["environments"][0].closed
    decision = slot["predictions"][0]
    assert (
        len(decision["features"]) == 296
        and decision["features"][:148] == decision["features"][148:]
    )
    work = report["work"]
    assert work["live_prediction_calls_attempted"] == 1
    assert work["live_prediction_calls_returned"] == (0 if fault == "prediction" else 1)
    assert work["environment_steps_attempted"] == (
        1 if fault in ("step", "close_after_step") else 0
    )
    assert (
        work["environment_steps_returned"] == 0 and work["failure_work_may_be_unobserved"] is True
    )
    assert work["history_episodes_started"] == work["history_live_feature_rows"] == 1
    assert work["history_live_feature_array_bytes"] == 296 * 8
    assert case.budget.updates_completed == 0
    if fault == "close_after_step":
        assert "native step failed" in report["reason"] and "close failed" in slot["close_error"]


@pytest.mark.parametrize("boundary", ["completed", "budget"])
def test_should_check_original_budget_and_completed_evidence_before_any_fake_d_work(
    monkeypatch, history_case, boundary
):
    case = history_case

    def refuse():
        raise ValueError("original completed boundary refused")

    if boundary == "budget":
        monkeypatch.setattr(case.budget, "before_final", refuse)
    else:
        case.work["on_guard"] = refuse
    with pytest.raises(evaluation.MiniGridEvaluationFailed, match="original completed boundary"):
        readout(case, final=True, public_history=True)
    assert not case.work.get("forks") and not case.work.get("environments")
    assert case.budget.updates_completed == 0
