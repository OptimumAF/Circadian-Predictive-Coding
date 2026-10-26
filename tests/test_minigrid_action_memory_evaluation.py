"""Real count-memory frozen readouts with array fixtures and a fake environment.

Fake environment success is a harness fact, never MiniGrid learning evidence.
There is no native environment, teacher, neural learner or optimizer here.
"""

import builtins
from copy import deepcopy
from dataclasses import replace
import pickle
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.adapters.action_memory_learner import ActionMemoryLearner
from src.adapters import minigrid_policy_evaluation as evaluation
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.action_memory import ActionMemoryPolicy
from src.core.demonstration_episode import episode_arrays
from src.core.minigrid_key_door import predefined_layouts
from src.core.minigrid_public_history import encode_public_history_episode
from test_minigrid_history_evaluation import CueFrameEnvironment
from test_minigrid_learning_pilot import Environment, source_batch


POINTS = (0, 32, 64, 96, 128)


def memory_policy() -> ActionMemoryPolicy:
    return ActionMemoryPolicy(feature_dim=296, num_classes=7, max_payload_bytes=155136)


@pytest.fixture(scope="module")
def engineering_states():
    """Generate clock-valid states once from deterministic nine-row TRAIN arrays."""
    sources = tuple(
        source_batch(context, cue) for context in ("A", "B", "C_train") for cue in ("blue", "red")
    )
    learner = ActionMemoryLearner(memory_policy())
    states = {0: learner.snapshot_state()}
    for update in range(1, 129):
        episode = sources[(update - 1) % len(sources)].episode
        assert episode is not None
        x, y = episode_arrays(encode_public_history_episode(episode))
        learner.train_batch(x, y)
        if update in POINTS:
            states[update] = learner.snapshot_state()
    return sources, states


@pytest.fixture
def memory_case(monkeypatch, engineering_states):
    original_sources, original_states = engineering_states
    work: dict[str, Any] = {}
    monkeypatch.setattr(Environment, "work", work)
    monkeypatch.setattr(evaluation, "MiniGridKeyDoor", CueFrameEnvironment)
    original_predict = ActionMemoryLearner.predict
    original_fork = ActionMemoryLearner.fork
    original_import = builtins.__import__
    source = ActionMemoryLearner(memory_policy())
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=128), lambda: 10.0)
    for _ in range(128):
        budget.record_update()
    binding = (budget.budget, budget.started_at, budget.updates_completed)

    def forbid_neural_comparison(*args, **kwargs):
        pytest.fail("memory evaluator used its neural full-state comparator")

    def no_torch(name: str, *args: Any, **options: Any) -> Any:
        if name == "torch" or name.startswith("torch."):
            pytest.fail("memory evaluator imported Torch")
        return original_import(name, *args, **options)

    def forbid_training(*args, **kwargs):
        pytest.fail("a frozen readout attempted storage updates")

    def fork(self):
        work.setdefault("forks", []).append(self)
        if work.get("identity_fork"):
            return self
        result = original_fork(self)
        work.setdefault("scorers", []).append(result)
        return result

    def predict(self, features):
        work.setdefault("attempts", []).append((self, features.copy()))
        live = len(features) == 1
        if live and work.get("prediction_error"):
            raise RuntimeError("memory live prediction failed")
        result = original_predict(self, features)
        work.setdefault("predicts", []).append((self, features.copy(), result.copy()))
        callback = work.get("on_predict")
        if callback is not None:
            callback(self, features)
        if live and work.get("bad_probability"):
            return np.ones((1, 6), dtype=np.float64)
        return result

    def guard():
        work.setdefault("guard_calls", []).append(len(work.get("predicts", [])))
        callback = work.get("on_guard")
        if callback is not None:
            callback()
        if (budget.budget, budget.started_at, budget.updates_completed) != binding:
            raise ValueError("original memory budget binding changed")

    monkeypatch.setattr(ActionMemoryLearner, "fork", fork)
    monkeypatch.setattr(ActionMemoryLearner, "predict", predict)
    monkeypatch.setattr(ActionMemoryLearner, "train_batch", forbid_training)
    monkeypatch.setattr(evaluation, "_same_state", forbid_neural_comparison)
    monkeypatch.setattr(builtins, "__import__", no_torch)
    return SimpleNamespace(
        source=source,
        sources=deepcopy(original_sources),
        states=deepcopy(original_states),
        budget=budget,
        binding=binding,
        work=work,
        guard=guard,
    )


def evaluate(case, *, updates=128, state=None, **options):
    defaults: dict[str, Any] = dict(
        method="observation_memory",
        updates=updates,
        layouts=predefined_layouts()[:3],
        role="train",
        sources=case.sources,
        budget=case.budget,
        check_boundary=case.guard,
        public_history=True,
    )
    defaults.update(options)
    return evaluation.evaluate_checkpoint(
        case.source, case.states[updates] if state is None else state, **defaults
    )


@pytest.mark.parametrize("updates", POINTS)
def test_should_read_clock_valid_owned_memory_states_without_training_or_renewing_budget(
    memory_case, updates
):
    case = memory_case
    state = case.states[updates]
    before = pickle.dumps((case.source.snapshot_state(), state, case.sources))
    report = evaluate(case, updates=updates)
    assert report["status"] == "complete" and report["state_unchanged"] is True
    assert report["method"] == "observation_memory" and report["updates"] == updates
    assert (state.completed_updates, state.training_examples) == (updates, 9 * updates)
    assert pickle.dumps((case.source.snapshot_state(), state, case.sources)) == before
    assert (
        case.budget.budget,
        case.budget.started_at,
        case.budget.updates_completed,
    ) == case.binding
    assert len(case.work["forks"]) == 1 and case.work["scorers"][0] is not case.source
    assert case.work["scorers"][0].snapshot_state() == state
    assert len(report["scores"]) == len(report["episodes"]) == 6
    assert [(score["context"], score["cue"]) for score in report["scores"]] == [
        (context, cue) for context in ("A", "B", "C_train") for cue in ("blue", "red")
    ]
    for index, packet in enumerate(case.sources):
        assert packet.episode is not None
        expected, _ = episode_arrays(encode_public_history_episode(packet.episode))
        owner, actual, probabilities = case.work["predicts"][index]
        assert owner is case.work["scorers"][0]
        np.testing.assert_array_equal(actual, expected)
        assert actual.shape == (9, 296) and not np.shares_memory(actual, packet.features)
        assert np.array(report["scores"][index]["probabilities"]).shape == (9, 7)
        if updates == 0:
            np.testing.assert_array_equal(probabilities, np.full((9, 7), 1 / 7))
            assert report["scores"][index]["ce"] == pytest.approx(np.log(7))
            assert report["scores"][index]["accuracy"] == 0
        else:
            assert report["scores"][index]["ce"] == 0
            assert (
                report["scores"][index]["accuracy"]
                == report["scores"][index]["exact_sequence_accuracy"]
                == 1
            )
    for index, (game, slot) in enumerate(
        zip(case.work["environments"], report["episodes"], strict=True)
    ):
        assert game.closed and game.role == slot["source_role"] == "train"
        assert game.layout == predefined_layouts()[index // 2]
        assert game.actions == [0, 0]
        initial = game.transitions[0]["observation"]
        for step, decision in enumerate(slot["predictions"]):
            expected = initial + game.transitions[step]["observation"]
            assert tuple(decision["features"]) == expected
            np.testing.assert_array_equal(
                case.work["predicts"][6 + 2 * index + step][1], [expected]
            )
            np.testing.assert_array_equal(decision["probabilities"], [1 / 7] * 7)
            assert decision["action"] == 0
    assert (
        case.work["environments"][0].transitions[0]["observation"]
        != case.work["environments"][1].transitions[0]["observation"]
    )
    work = report["work"]
    assert work["raw_score_forwards_completed"] == 6 and work["raw_score_rows_completed"] == 54
    assert work["live_prediction_calls_returned"] == 12 and work["live_sessions_started"] == 0
    assert work["environment_resets_completed"] == 6 and work["environment_steps_returned"] == 12
    assert work["history_raw_episodes_encoded"] == work["history_episodes_started"] == 6
    assert work["history_live_feature_rows"] == 12
    assert work["history_raw_feature_array_bytes"] == 6 * 9 * 296 * 8
    assert work["history_live_feature_array_bytes"] == 12 * 296 * 8


@pytest.mark.parametrize(
    "fault",
    [
        "history_false",
        "history_bool",
        "final",
        "probe",
        "a_only",
        "extra_layout",
        "changed_layout",
        "layout_extra",
        "source_missing",
        "source_order",
        "source_role",
        "source_already_encoded",
        "source_mismatch",
        "updates_bool",
        "undeclared_point",
        "old_policy",
        "small_cap",
        "wrong_updates",
        "wrong_rows",
        "native_override",
    ],
)
def test_should_refuse_wrong_memory_routes_geometries_sources_or_clocks_before_any_work(
    memory_case, fault
):
    case = memory_case
    state: Any = case.states[128]
    options: dict[str, Any] = {}
    if fault == "history_false":
        options["public_history"] = False
    elif fault == "history_bool":
        options["public_history"] = np.bool_(True)
    elif fault == "final":
        options.update(role="final_test", layouts=predefined_layouts()[-1:], sources=())
    elif fault == "probe":
        options["role"] = "probe_test"
    elif fault == "a_only":
        options.update(layouts=predefined_layouts()[:1], sources=case.sources[:2])
    elif fault == "extra_layout":
        options["layouts"] = predefined_layouts()
    elif fault == "changed_layout":
        options["layouts"] = (
            replace(predefined_layouts()[0], wall_color="blue"),
            *predefined_layouts()[1:3],
        )
    elif fault == "layout_extra":
        layouts = deepcopy(predefined_layouts()[:3])
        object.__setattr__(layouts[0], "extra", 1)
        options["layouts"] = layouts
    elif fault == "source_missing":
        options["sources"] = case.sources[:-1]
    elif fault == "source_order":
        options["sources"] = tuple(reversed(case.sources))
    elif fault == "source_role":
        first = case.sources[0]
        assert first.episode is not None
        options["sources"] = (
            replace(first, role="probe_train", episode=replace(first.episode, role="probe_train")),
            *case.sources[1:],
        )
    elif fault == "source_already_encoded":
        first = case.sources[0]
        assert first.episode is not None
        episode = encode_public_history_episode(first.episode)
        x, y = episode_arrays(episode)
        options["sources"] = (
            replace(first, features=x, targets=y, episode=episode),
            *case.sources[1:],
        )
    elif fault == "source_mismatch":
        case.sources[0].features[0, 0] = 0.25
    elif fault in ("updates_bool", "undeclared_point"):
        options["updates"] = True if fault == "updates_bool" else 1
    elif fault in ("old_policy", "small_cap"):
        declared = (
            ActionMemoryPolicy()
            if fault == "old_policy"
            else ActionMemoryPolicy(feature_dim=296, num_classes=7)
        )
        state = ActionMemoryLearner(declared).snapshot_state()
    elif fault == "wrong_updates":
        state = replace(state, completed_updates=127)
    elif fault == "wrong_rows":
        state = replace(state, training_examples=1151)
    else:
        options["native_training_examples"] = 1152
    with pytest.raises(ValueError):
        evaluate(case, state=state, **options)
    assert not case.work.get("forks") and not case.work.get("predicts")
    assert not case.work.get("environments") and not case.work.get("guard_calls")
    assert case.budget.updates_completed == 128


@pytest.mark.parametrize("target", ["source", "scorer", "checkpoint", "budget", "clock"])
def test_should_stop_after_the_first_paid_raw_score_if_original_evidence_changes(
    memory_case, target
):
    case = memory_case
    state = case.states[128]

    def mutate(owner, features):
        if target == "source":
            case.source.restore_state(case.states[32])
        elif target == "scorer":
            owner.restore_state(case.states[32])
        elif target == "checkpoint":
            object.__setattr__(state, "completed_updates", 127)
        elif target == "budget":
            case.budget.record_update()
        else:
            case.budget.last_clock = 11.0

    case.work["on_predict"] = mutate
    with pytest.raises(evaluation.MiniGridEvaluationFailed) as raised:
        evaluate(case, state=state)
    report = raised.value.report
    assert report["status"] == "failed" and report["state_unchanged"] is False
    assert (
        report["work"]["raw_score_forwards_attempted"]
        == report["work"]["raw_score_forwards_completed"]
        == 1
    )
    assert report["work"]["raw_score_rows_completed"] == 9
    assert not report["scores"] and not report["episodes"] and not case.work.get("environments")
    assert case.budget.updates_completed == (129 if target == "budget" else 128)


@pytest.mark.parametrize("boundary", ["guard", "budget", "identity_fork"])
def test_should_refuse_failed_completed_boundary_or_source_alias_before_predictions(
    memory_case, monkeypatch, boundary
):
    case = memory_case

    def refuse():
        raise ValueError("original memory boundary refused")

    if boundary == "guard":
        case.work["on_guard"] = refuse
    elif boundary == "budget":
        monkeypatch.setattr(case.budget, "before_final", refuse)
    else:
        case.work["identity_fork"] = True
    with pytest.raises(evaluation.MiniGridEvaluationFailed):
        evaluate(case)
    assert not case.work.get("predicts") and not case.work.get("environments")
    assert case.budget.updates_completed == 128


@pytest.mark.parametrize(
    "fault", ["prediction_error", "bad_probability", "step_error", "step_and_close"]
)
def test_should_preserve_raw_scores_and_paid_live_prefix_when_work_fails(memory_case, fault):
    case = memory_case
    case.work["step_error" if fault == "step_and_close" else fault] = True
    if fault == "step_and_close":
        case.work["close_error"] = True
    with pytest.raises(evaluation.MiniGridEvaluationFailed) as raised:
        evaluate(case)
    report = raised.value.report
    assert len(report["scores"]) == 6 and len(report["episodes"]) == 1
    slot = report["episodes"][0]
    assert slot["status"] == "failed" and len(slot["predictions"]) == 1
    assert case.work["environments"][0].closed and len(case.work["environments"]) == 1
    decision = slot["predictions"][0]
    assert (
        len(decision["features"]) == 296
        and decision["features"][:148] == decision["features"][148:]
    )
    work = report["work"]
    assert work["raw_score_forwards_completed"] == 6 and work["raw_score_rows_completed"] == 54
    assert work["live_prediction_calls_attempted"] == 1
    assert work["live_prediction_calls_returned"] == (0 if fault == "prediction_error" else 1)
    assert work["environment_steps_attempted"] == (
        1 if fault in ("step_error", "step_and_close") else 0
    )
    assert (
        work["environment_steps_returned"] == 0 and work["failure_work_may_be_unobserved"] is True
    )
    assert (
        work["history_live_feature_rows"] == 1
        and work["history_live_feature_array_bytes"] == 296 * 8
    )
    assert case.budget.updates_completed == 128
    if fault == "step_and_close":
        assert "native step failed" in report["reason"] and "close failed" in slot["close_error"]


def test_should_detect_checkpoint_mutation_during_source_snapshot_readback(
    memory_case, monkeypatch
):
    case = memory_case
    state = case.states[128]
    original_snapshot = ActionMemoryLearner.snapshot_state

    def snapshot(owner):
        result = original_snapshot(owner)
        if owner is case.source and case.work.get("predicts"):
            object.__setattr__(state, "completed_updates", 127)
        return result

    monkeypatch.setattr(ActionMemoryLearner, "snapshot_state", snapshot)
    with pytest.raises(evaluation.MiniGridEvaluationFailed, match="checkpoint changed") as raised:
        evaluate(case, state=state)
    assert raised.value.report["work"]["raw_score_forwards_completed"] == 1
    assert len(case.work["predicts"]) == 1 and not case.work.get("environments")


def test_should_preserve_prediction_costs_and_unknown_environment_work_when_readback_fails(
    memory_case, monkeypatch
):
    case = memory_case
    descriptor: Any = vars(Environment)["report"]
    original_report = descriptor.fget
    assert original_report is not None

    def report_after_close(game):
        if game.closed:
            raise RuntimeError("environment readback failed after close")
        return original_report(game)

    monkeypatch.setattr(Environment, "report", property(report_after_close))
    with pytest.raises(evaluation.MiniGridEvaluationFailed) as raised:
        evaluate(case)
    report = raised.value.report
    assert len(report["scores"]) == 6 and len(report["episodes"]) == 1
    assert "readback failed" in report["episodes"][0]["readback_error"]
    assert report["work"]["live_prediction_calls_returned"] == 2
    assert report["work"]["failure_work_may_be_unobserved"] is True
    assert len(case.work["environments"]) == 1 and case.budget.updates_completed == 128
