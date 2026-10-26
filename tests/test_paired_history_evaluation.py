"""Frozen paired history readouts retain public inputs and fresh RNN sessions.

Only the native snapshot/tensor boundary is replaced. The existing evaluator,
encoder, scoring, lifecycle, callbacks and budgets execute with owned fake ports.
No optional runtime, update, environment physics or scientific comparison occurs.
"""

from copy import deepcopy
from dataclasses import fields, replace
import pickle
from typing import Any

import numpy as np
import pytest

from src.adapters import minigrid_policy_evaluation as evaluation
from src.adapters import torch_recurrent_episode_batch as paired
from src.adapters import torch_recurrent_learner as native
from src.core.demonstration_episode import episode_arrays
from src.core.minigrid_key_door import predefined_layouts
from src.core.minigrid_public_history import encode_public_history_episode
from test_minigrid_history_evaluation import history_case as history_case
from test_minigrid_learning_pilot import Learner, evaluator as evaluator

METHOD = "recurrent_paired_bc"


@pytest.fixture
def paired_case(monkeypatch, history_case):
    case = history_case
    policy = native.TorchRecurrentPolicy(296, 8, 7, 17)
    case.source.policy = policy
    case.source.state = paired.TorchRecurrentEpisodeBatchSnapshot(
        1,
        policy,
        {"weight": np.zeros((1, 1), dtype=np.float32)},
        {"momentum": 0.0},
        {"weight": None},
        0,
        0,
        0,
    )
    case.sources = case.sources[:2]
    calls: list[Any] = []

    def snapshot(state):
        # The real native boundary's tensor/optimizer law has independent native
        # qualification. Keep its record/policy/count admission while substituting
        # tiny NumPy payloads solely for these environment/callback tests.
        if type(state) is not paired.TorchRecurrentEpisodeBatchSnapshot or set(vars(state)) != {
            field.name for field in fields(type(state))
        }:
            raise ValueError("exact paired native snapshot required")
        native._policy(state.policy)
        if any(
            type(getattr(state, name)) is not int or getattr(state, name) < 0
            for name in ("completed_updates", "training_episodes", "training_examples")
        ):
            raise ValueError("native integer paired clocks required")
        calls.append(state)
        return deepcopy(state)

    monkeypatch.setattr(paired, "_snapshot", snapshot)
    case.native_validations = calls
    return case


def checkpoint(case, point=32, width=296):
    policy = replace(case.source.policy, feature_dim=width)
    case.source.policy = policy
    case.source.state = replace(case.source.state, policy=policy)
    return replace(
        case.source.snapshot_state(),
        completed_updates=point,
        training_episodes=2 * point,
        training_examples=18 * point,
    )


def readout(case, *, state=None, point=32, **options):
    if state is None:
        state = checkpoint(case, point)
    arguments: dict[str, Any] = dict(
        method=METHOD,
        updates=point,
        layouts=predefined_layouts()[:1],
        role="train",
        sources=case.sources,
        budget=case.budget,
        check_boundary=case.guard,
        public_history=True,
    )
    arguments.update(options)
    return evaluation.evaluate_checkpoint(case.source, state, **arguments)


@pytest.mark.parametrize("point", [0, 64])
def test_should_encode_each_cue_causally_and_keep_fresh_recurrent_activation(paired_case, point):
    case = paired_case
    state = checkpoint(case, point)
    frozen = pickle.dumps((case.sources, case.source.snapshot_state(), state))
    binding = (
        case.budget.budget,
        case.budget.clock,
        case.budget.started_at,
        case.budget.updates_completed,
    )
    report = readout(case, state=state, point=point)
    assert report["status"] == "complete" and report["state_unchanged"] is True
    assert report["method"] == METHOD and report["role"] == "train"
    assert case.native_validations and not case.work.get("trains")
    assert pickle.dumps((case.sources, case.source.snapshot_state(), state)) == frozen
    assert (
        case.budget.budget,
        case.budget.clock,
        case.budget.started_at,
        case.budget.updates_completed,
    ) == binding
    assert len(case.work["predicts"]) == 6
    assert len(case.work["sessions"]) == 2
    assert case.work["sessions"][0] is not case.work["sessions"][1]
    assert all(session.completed_steps == 2 for session in case.work["sessions"])
    assert [(score["context"], score["cue"]) for score in report["scores"]] == [
        ("A", "blue"),
        ("A", "red"),
    ]
    for index, packet in enumerate(case.sources):
        assert packet.episode is not None
        expected, _ = episode_arrays(encode_public_history_episode(packet.episode))
        actual = case.work["predicts"][index][1]
        np.testing.assert_array_equal(actual, expected)
        assert actual.shape == (9, 296) and packet.features.shape == (9, 148)
        assert not np.shares_memory(actual, packet.features)
    initials = []
    for index, (game, slot) in enumerate(
        zip(case.work["environments"], report["episodes"], strict=True)
    ):
        assert game.closed and game.actions == [0, 0]
        assert slot["source_role"] == "train" and slot["state_unchanged"] is True
        initial = game.transitions[0]["observation"]
        initials.append(initial)
        assert slot["success"] is True and slot["return"] == 0.5
        for step, decision in enumerate(slot["predictions"]):
            current = game.transitions[step]["observation"]
            assert len(decision["observation"]) == 148
            assert tuple(decision["features"]) == initial + current
            assert tuple(decision["observation"]) == current
            np.testing.assert_array_equal(
                case.work["predicts"][2 + 2 * index + step][1], [initial + current]
            )
            assert decision["probabilities"] == [0.5, 0.5, 0, 0, 0, 0, 0]
            assert decision["action"] == 0
            assert not any(name in decision for name in ("target", "cue", "context", "reward"))
        assert slot["predictions"][1]["features"][:148] != slot["predictions"][1]["observation"]
    assert initials[0] != initials[1]
    work = report["work"]
    assert (work["raw_score_forwards_completed"], work["raw_score_rows_completed"]) == (2, 18)
    assert (
        work["live_prediction_calls_returned"],
        work["live_sessions_started"],
        work["environment_resets_completed"],
        work["environment_steps_returned"],
    ) == (4, 2, 2, 4)
    assert work["history_episodes_started"] == work["history_raw_episodes_encoded"] == 2
    assert work["history_live_feature_rows"] == 4
    assert work["history_live_feature_array_bytes"] == 4 * 296 * 8
    assert work["history_raw_feature_array_bytes"] == 2 * 9 * 296 * 8
    assert work["history_raw_target_array_bytes"] == 2 * 9 * 8
    assert report["public_history"]["applied"] is True
    assert report["public_history"]["prediction_feature_dim"] == 296
    assert report["public_history"]["initial_numeric_bytes"] == 1184


def test_should_preserve_raw_paired_default_and_explicit_false_wire(paired_case):
    case = paired_case
    state = checkpoint(case, width=148)
    options = dict(
        method=METHOD,
        updates=32,
        layouts=predefined_layouts()[:1],
        role="train",
        sources=case.sources,
        budget=case.budget,
        check_boundary=case.guard,
    )
    default = evaluation.evaluate_checkpoint(case.source, state, **options)
    explicit = evaluation.evaluate_checkpoint(case.source, state, **options, public_history=False)
    assert default == explicit and "public_history" not in default
    assert not any(name.startswith("history_") for name in default["work"])
    assert all(features.shape[1] == 148 for _, features in case.work["predicts"])
    assert all(
        "features" not in decision
        for slot in default["episodes"]
        for decision in slot["predictions"]
    )
    assert not case.work.get("trains")


@pytest.mark.parametrize(
    "fault",
    [
        "raw_width",
        "hidden",
        "seed",
        "rate",
        "steps",
        "episodes",
        "rows",
        "bool_count",
        "legacy",
        "extra",
    ],
)
def test_should_validate_paired_history_header_before_fork_or_raw_prediction(paired_case, fault):
    case = paired_case
    state: Any = checkpoint(case)
    if fault == "raw_width":
        state = replace(state, policy=replace(state.policy, feature_dim=148))
    elif fault == "hidden":
        state = replace(state, policy=replace(state.policy, hidden_dim=4))
    elif fault == "seed":
        state = replace(state, policy=replace(state.policy, seed=18))
    elif fault == "rate":
        state = replace(state, policy=replace(state.policy, learning_rate=0.04))
    elif fault == "steps":
        state = replace(state, completed_updates=31)
    elif fault == "episodes":
        state = replace(state, training_episodes=63)
    elif fault == "rows":
        state = replace(state, training_examples=575)
    elif fault == "bool_count":
        state = replace(state, training_episodes=True)
    elif fault == "legacy":
        state = native.TorchRecurrentSnapshot(
            1,
            state.policy,
            state.parameters,
            state.optimizer_state,
            state.gradients,
            state.completed_updates,
            state.training_examples,
        )
    else:
        object.__setattr__(state, "unexpected", True)
    with pytest.raises((ValueError, evaluation.MiniGridEvaluationFailed)):
        readout(case, state=state)
    assert not case.work.get("forks") and not case.work.get("predicts")
    assert not case.work.get("environments") and case.budget.updates_completed == 0


@pytest.mark.parametrize(
    "fault",
    [
        "point",
        "point_bool",
        "final",
        "probe",
        "geometry",
        "other_layout",
        "order",
        "missing",
        "encoded_sources",
        "source_role",
        "row_override",
    ],
)
def test_should_keep_exact_train_a_original_source_admission_before_work(paired_case, fault):
    case = paired_case
    state = checkpoint(case)
    options: dict[str, Any] = {}
    if fault == "point":
        options["updates"] = 8
    elif fault == "point_bool":
        options["updates"] = True
    elif fault == "final":
        options.update(role="final_test", layouts=predefined_layouts()[-1:], sources=())
    elif fault == "probe":
        options["role"] = "probe_test"
    elif fault == "geometry":
        options["layouts"] = (replace(predefined_layouts()[0], rotation=1),)
    elif fault == "other_layout":
        options["layouts"] = predefined_layouts()[1:2]
    elif fault == "order":
        options["sources"] = tuple(reversed(case.sources))
    elif fault == "missing":
        options["sources"] = case.sources[:1]
    elif fault == "row_override":
        options["native_training_examples"] = 576
    else:
        first, second = case.sources
        assert first.episode is not None
        if fault == "encoded_sources":
            episode = encode_public_history_episode(first.episode)
            features, targets = episode_arrays(episode)
            first = replace(first, episode=episode, features=features, targets=targets)
        else:
            first = replace(
                first, role="probe_train", episode=replace(first.episode, role="probe_train")
            )
        options["sources"] = (first, second)
    with pytest.raises(ValueError):
        readout(case, state=state, **options)
    assert not case.work.get("forks") and not case.work.get("predicts")
    assert not case.work.get("environments")


@pytest.mark.parametrize("target", ["source", "scorer", "optimizer", "checkpoint"])
def test_should_refuse_state_changes_after_paid_raw_history_prediction(paired_case, target):
    case = paired_case
    state = checkpoint(case)

    def mutate(owner):
        if target == "checkpoint":
            state.parameters["weight"][0, 0] += 1
        elif target == "optimizer":
            owner.state.optimizer_state["momentum"] += 1
        else:
            selected = case.source if target == "source" else owner
            selected.state.parameters["weight"][0, 0] += 1

    case.work["on_predict"] = mutate
    with pytest.raises(evaluation.MiniGridEvaluationFailed) as raised:
        readout(case, state=state)
    report = raised.value.report
    assert report["work"]["raw_score_forwards_completed"] == 1
    assert report["work"]["raw_score_rows_completed"] == 9
    assert report["work"]["history_raw_feature_array_bytes"] == 9 * 296 * 8
    assert len(case.work["predicts"]) == 1 and not case.work.get("environments")
    assert case.budget.updates_completed == 0


@pytest.mark.parametrize("fault", ["initial_steps", "reused", "state", "double_step"])
def test_should_require_owned_clean_sessions_and_exactly_one_activation_advance(
    paired_case,
    monkeypatch,
    fault,
):
    case = paired_case
    start = Learner.start_episode

    def session(owner):
        result = start(owner)
        if fault == "initial_steps":
            result.completed_steps = 1
        elif fault == "reused" and len(case.work["sessions"]) > 1:
            return case.work["sessions"][0]
        return result

    monkeypatch.setattr(Learner, "start_episode", session)
    if fault in ("state", "double_step"):

        def mutate(current):
            if fault == "state":
                current.state.parameters["weight"][0, 0] += 1
            else:
                current.completed_steps += 1

        case.work["on_session_predict"] = mutate
    with pytest.raises(evaluation.MiniGridEvaluationFailed) as raised:
        readout(case)
    report = raised.value.report
    assert report["work"]["raw_score_forwards_completed"] == 2
    assert report["work"]["raw_score_rows_completed"] == 18
    assert not case.work.get("trains") and case.budget.updates_completed == 0
    if fault == "initial_steps":
        assert not case.work.get("environments")
    elif fault == "reused":
        assert len(case.work["environments"]) == 1
        assert report["episodes"][0]["status"] == "complete"
    else:
        assert len(case.work["environments"]) == 1
        assert case.work["environments"][0].closed
        assert report["work"]["live_prediction_calls_returned"] == 1


@pytest.mark.parametrize("fault", ["step_and_close", "close_only"])
def test_should_retain_paid_history_prefix_and_primary_cleanup_failure(paired_case, fault):
    case = paired_case
    case.work["close_error"] = True
    if fault == "step_and_close":
        case.work["step_error"] = True
    with pytest.raises(evaluation.MiniGridEvaluationFailed) as raised:
        readout(case)
    report = raised.value.report
    slot = report["episodes"][0]
    assert len(report["episodes"]) == 1 and len(case.work["environments"]) == 1
    assert case.work["environments"][0].closed and "close failed" in slot["close_error"]
    assert len(slot["predictions"][0]["features"]) == 296
    assert report["work"]["raw_score_rows_completed"] == 18
    assert report["work"]["environment_steps_attempted"] == (1 if fault == "step_and_close" else 2)
    assert report["work"]["environment_steps_returned"] == (0 if fault == "step_and_close" else 2)
    if fault == "step_and_close":
        assert "native step failed" in report["reason"]
    else:
        assert "close failed" in report["reason"]
    assert case.budget.updates_completed == 0


@pytest.mark.parametrize("kind", ["budget", "completed_receipts"])
def test_should_observe_original_budget_and_preceding_completion_before_any_work(
    paired_case, monkeypatch, kind
):
    case = paired_case
    state = checkpoint(case)

    def refuse():
        raise ValueError("original paid boundary refused")

    if kind == "budget":
        monkeypatch.setattr(case.budget, "before_final", refuse)
    else:
        case.work["on_guard"] = refuse
    with pytest.raises(ValueError, match="original paid boundary"):
        readout(case, state=state)
    assert not case.work.get("forks") and not case.work.get("predicts")
    assert not case.work.get("environments") and case.budget.updates_completed == 0


def test_should_refuse_final_scorer_snapshot_callback_changing_supplied_checkpoint(
    paired_case,
    monkeypatch,
):
    case = paired_case
    state = checkpoint(case)
    installed = fired = False

    def install(owner):
        nonlocal installed
        if installed:
            return
        installed = True
        snapshot = owner.snapshot_state

        def readback():
            nonlocal fired
            observed = snapshot()
            games = case.work.get("environments", [])
            if not fired and len(games) == 2 and all(game.closed for game in games):
                fired = True
                state.parameters["weight"][0, 0] = 9.0
            return observed

        monkeypatch.setattr(owner, "snapshot_state", readback)

    case.work["on_predict"] = install
    with pytest.raises(evaluation.MiniGridEvaluationFailed) as raised:
        readout(case, state=state)
    report = raised.value.report
    assert fired and report["status"] == "failed"
    assert state.parameters["weight"][0, 0] == 9.0
    assert len(report["scores"]) == len(report["episodes"]) == 2
    assert report["work"]["raw_score_rows_completed"] == 18
    assert report["work"]["live_prediction_calls_returned"] == 4
    assert report["work"]["environment_steps_returned"] == 4
    assert all(game.closed for game in case.work["environments"])
    assert not case.work.get("trains") and case.budget.updates_completed == 0
