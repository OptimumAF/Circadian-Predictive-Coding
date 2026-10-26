"""Recurring invariant evaluator ports without count prediction or physics.

Literal typed snapshots and scripted seven-class probabilities test admission,
causal history, full-state guards and accounting. Scripted game endings do not
establish native skill, adaptation or correctness of MiniGrid dynamics.
"""

import builtins
from copy import deepcopy
from dataclasses import replace
import pickle
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.adapters import minigrid_policy_evaluation as evaluation
from src.adapters.action_memory_learner import ActionMemoryLearner
from src.adapters.fifo_action_memory_learner import FifoActionMemoryLearner
from src.adapters.invariant_action_memory_learner import (
    InvariantActionMemoryLearner,
    InvariantMemoryPolicy,
    InvariantMemorySnapshot,
    copy_invariant_memory_snapshot,
)
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.action_memory import ActionMemoryEntry, ActionMemoryPolicy, ActionMemorySnapshot
from src.core.demonstration_episode import episode_arrays
from src.core.minigrid_key_door import predefined_layouts
from src.core.minigrid_invariant_encoding import INVARIANT_HISTORY_ENCODING_ID
from src.core.minigrid_public_history import encode_public_history_episode
from test_minigrid_learning_pilot import source_batch
from test_minigrid_probe_live_evaluator import ScriptedEnvironment, snapshot as other_snapshot

POINTS = (0, 32, 64, 96, 128)


def snapshot(updates=0, *, cap=64, payload=155136, rows=None):
    native_policy = ActionMemoryPolicy(
        feature_dim=296, num_classes=7, max_entries=cap, max_payload_bytes=payload
    )
    rows = 9 * updates if rows is None else rows
    entries = () if rows == 0 else (ActionMemoryEntry((-1.0,) * 296, (rows, 0, 0, 0, 0, 0, 0)),)
    return InvariantMemorySnapshot(
        InvariantMemoryPolicy(native_policy),
        ActionMemorySnapshot(native_policy, entries, updates, rows),
    )


class ScriptedCountPort:
    def __init__(self, state, work, *, scorer=False):
        self.value = copy_invariant_memory_snapshot(state)
        self.work = work
        self.scorer = scorer

    def snapshot_state(self):
        hook = self.work.get("on_snapshot")
        if hook is not None:
            hook(self)
        return deepcopy(self.value)

    def fork(self):
        self.work.setdefault("forks", []).append(self)
        if self.work.get("identity_fork"):
            return self
        child = ScriptedCountPort(self.value, self.work, scorer=True)
        self.work["scorer"] = child
        return child

    def restore_state(self, state):
        self.value = copy_invariant_memory_snapshot(state)
        self.work.setdefault("restores", []).append(self)

    def predict(self, features):
        assert features.dtype == np.float64 and features.shape in ((9, 296), (1, 296))
        self.work.setdefault("attempts", []).append(features.copy())
        if self.work.get("prediction_error"):
            raise RuntimeError("scripted recurring predictor failed")
        result = np.full((len(features), 7), 1 / 7, dtype=np.float64)
        self.work.setdefault("returns", []).append(features.copy())
        hook = self.work.get("on_predict")
        if hook is not None:
            hook(self, features)
        if self.work.get("bad_probability"):
            result[0, 0] = np.nan
        return result

    def train_batch(self, *args):
        pytest.fail("recurring frozen evaluator reached training")


@pytest.fixture
def case(monkeypatch):
    work: dict[str, Any] = {}
    monkeypatch.setattr(ScriptedEnvironment, "work", work)
    monkeypatch.setattr(evaluation, "MiniGridKeyDoor", ScriptedEnvironment)

    def forbidden(*args, **options):
        pytest.fail("scripted boundary test reached a real learner or Torch comparison")

    for kind in (ActionMemoryLearner, FifoActionMemoryLearner, InvariantActionMemoryLearner):
        monkeypatch.setattr(kind, "predict", forbidden)
        monkeypatch.setattr(kind, "train_batch", forbidden)
    monkeypatch.setattr(evaluation, "_same_state", forbidden)
    original_import = builtins.__import__

    def no_torch(name, *args, **options):
        if name == "torch" or name.startswith("torch."):
            pytest.fail("invariant recurring readout imported Torch")
        return original_import(name, *args, **options)

    monkeypatch.setattr(builtins, "__import__", no_torch)
    sources = tuple(
        source_batch(context, cue) for context in ("A", "B", "C_train") for cue in ("blue", "red")
    )
    source = ScriptedCountPort(snapshot(), work)
    budget = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=128, max_replay_examples=0), lambda: 10.0
    )
    budget.updates_completed = 128  # Literal completed engineering exposure, no training.
    config, raw_inputs = budget.budget, pickle.dumps(sources)
    result = SimpleNamespace(
        source=source,
        sources=sources,
        state=snapshot(128),
        budget=budget,
        work=work,
        layouts=predefined_layouts()[:3],
    )

    def guard():
        work["guard_calls"] = work.get("guard_calls", 0) + 1
        hook = work.get("on_guard")
        if hook is not None:
            hook(result)
        if (
            budget.budget is not config
            or type(budget.updates_completed) is not int
            or budget.updates_completed != 128
            or budget.replay_examples_completed != 0
            or pickle.dumps(sources) != raw_inputs
        ):
            raise ValueError("original completed budget or TRAIN cohort changed")

    result.guard = guard
    return result


def evaluate(case, *, updates=128, state=None, **changes):
    options = dict(
        updates=updates,
        layouts=case.layouts,
        role="train",
        sources=case.sources,
        budget=case.budget,
        check_boundary=case.guard,
    )
    options.update(changes)
    return evaluation.evaluate_invariant_memory_checkpoint(
        case.source, case.state if state is None else state, **options
    )


@pytest.mark.parametrize("point", POINTS)
def test_should_use_all_original_points_and_count_raw_plus_live_projection_returns(case, point):
    state = snapshot(point)
    before = pickle.dumps((case.source.value, state, case.sources))
    report = evaluate(case, updates=point, state=state)
    assert report["status"] == "complete" and report["state_unchanged"] is True
    assert report["method"] == "public_invariant_observation_memory" and report["role"] == "train"
    assert report["model_encoding"] == INVARIANT_HISTORY_ENCODING_ID
    assert report["native_training_examples"] == 9 * point
    assert [(s["context"], s["cue"]) for s in report["scores"]] == [
        (context, cue) for context in ("A", "B", "C_train") for cue in ("blue", "red")
    ]
    assert [(s["context"], s["cue"], s["source_role"]) for s in report["episodes"]] == [
        (context, cue, "train") for context in ("A", "B", "C_train") for cue in ("blue", "red")
    ]
    work = report["work"]
    assert work["raw_score_forwards_completed"] == 6 and work["raw_score_rows_completed"] == 54
    assert work["live_prediction_calls_returned"] == work["environment_steps_returned"] == 12
    assert work["internal_projection_rows_returned"] == 66
    assert work["internal_projection_array_bytes_returned"] == 66 * 2368
    assert len(case.work["forks"]) == len(case.work["restores"]) == 1
    for batch, actual in zip(case.sources, case.work["returns"][:6], strict=True):
        expected, _ = episode_arrays(encode_public_history_episode(batch.episode))
        np.testing.assert_array_equal(actual, expected)
        assert not np.shares_memory(actual, batch.features)
    for episode in report["episodes"]:
        initial = episode["predictions"][0]["observation"]
        assert episode["success"] is True
        for decision in episode["predictions"]:
            assert decision["features"][:148] == initial
            assert decision["features"][148:] == decision["observation"]
            assert decision["action"] == 0
    assert pickle.dumps((case.source.value, state, case.sources)) == before
    assert case.budget.updates_completed == 128 and case.budget.last_clock == 10.0


@pytest.mark.parametrize(
    "fault",
    ["bool", "unlisted", "clock", "rows", "raw", "fifo", "cap", "payload", "encoding", "extra"],
)
def test_should_refuse_invalid_typed_clocks_and_policy_before_external_ports(case, fault):
    updates, state = 128, case.state
    if fault == "bool":
        updates = False
    elif fault == "unlisted":
        updates = 144
    elif fault == "clock":
        state = snapshot(96)
    elif fault == "rows":
        state = snapshot(128, rows=1151)
    elif fault == "raw":
        state = state.native_state
    elif fault == "fifo":
        state = other_snapshot(128, fifo=True)
    elif fault == "cap":
        state = snapshot(128, cap=63)
    elif fault == "payload":
        state = snapshot(128, payload=155135)
    elif fault == "encoding":
        object.__setattr__(state.policy, "encoding_id", "other")
    else:
        object.__setattr__(state, "extra", True)
    with pytest.raises(ValueError):
        evaluate(case, updates=updates, state=state)
    assert case.work == {}


@pytest.mark.parametrize(
    "fault",
    [
        "final",
        "probe",
        "A_only",
        "list",
        "layout",
        "extra_layout",
        "order",
        "history_sources",
        "source_role",
    ],
)
def test_should_reuse_the_original_exact_train_admission_before_work(case, fault):
    options: dict[str, Any] = {}
    if fault == "final":
        options.update(role="final_test", layouts=(predefined_layouts()[-1],), sources=())
    elif fault == "probe":
        options["role"] = "probe_test"
    elif fault == "A_only":
        options.update(layouts=case.layouts[:1], sources=case.sources[:2])
    elif fault == "list":
        options["layouts"] = list(case.layouts)
    elif fault == "layout":
        options["layouts"] = (replace(case.layouts[0], wall_color="red"), *case.layouts[1:])
    elif fault == "extra_layout":
        layout = replace(case.layouts[0])
        object.__setattr__(layout, "extra", 1)
        options["layouts"] = (layout, *case.layouts[1:])
    elif fault == "order":
        options["sources"] = tuple(reversed(case.sources))
    elif fault == "history_sources":
        options["sources"] = tuple(
            replace(batch, features=np.zeros((9, 296))) for batch in case.sources
        )
    else:
        options["sources"] = (replace(case.sources[0], role="final_test"), *case.sources[1:])
    with pytest.raises(ValueError):
        evaluate(case, **options)
    assert case.work == {}


@pytest.mark.parametrize("target", ["checkpoint", "source", "scorer", "budget", "cohort"])
def test_should_keep_returned_fit_projection_cost_before_a_late_guard_failure(case, target):
    def mutate(scorer, features):
        if target == "budget":
            case.budget.updates_completed = 127
        elif target == "cohort":
            case.sources[0].features[0, 0] = 0.25
        elif target == "source":
            case.source.value = snapshot(32)
        else:
            current = case.state if target == "checkpoint" else scorer.value
            entry = current.native_state.entries[0]
            object.__setattr__(entry, "counts", (entry.counts[0] - 1, 1, 0, 0, 0, 0, 0))

    case.work["on_predict"] = mutate
    with pytest.raises(evaluation.MiniGridEvaluationFailed) as caught:
        evaluate(case)
    report = caught.value.report
    assert report["status"] == "failed" and report["scores"] == []
    assert report["work"]["raw_score_forwards_completed"] == 1
    assert report["work"]["internal_projection_rows_returned"] == 9
    assert report["work"]["internal_projection_array_bytes_returned"] == 9 * 2368
    assert not case.work.get("environments")


def test_should_require_the_original_zero_source_on_every_snapshot_port(case):
    case.source.value = snapshot(128)
    with pytest.raises(evaluation.MiniGridEvaluationFailed) as caught:
        evaluate(case)
    assert "native0 source" in caught.value.report["reason"]
    assert not case.work.get("forks") and not case.work.get("attempts")


@pytest.mark.parametrize(
    "fault", ["prediction_error", "bad_probability", "step_error", "close_error", "readback_error"]
)
def test_should_preserve_partial_known_live_costs_and_no_further_games(case, fault):
    # Trigger after six successful raw scores so their paid rows always survive.
    def enable_after_fit(scorer, features):
        if len(features) == 1:
            case.work[fault] = True
        if fault == "prediction_error" and len(case.work["returns"]) == 6:
            case.work[fault] = True

    case.work["on_predict"] = enable_after_fit
    if fault == "prediction_error":

        def before_next(owner):
            if len(case.work.get("returns", [])) == 6:
                case.work[fault] = True

        case.work["on_guard"] = before_next
    with pytest.raises(evaluation.MiniGridEvaluationFailed) as caught:
        evaluate(case)
    report = caught.value.report
    assert len(report["scores"]) == 6
    assert report["work"]["raw_score_forwards_completed"] == 6
    assert report["work"]["internal_projection_rows_returned"] >= 54
    assert len(case.work["environments"]) == 1
    assert report["episodes"][0]["status"] == "failed"
    assert report["work"]["failure_work_may_be_unobserved"] is True


def test_should_refuse_a_changed_parent_after_the_last_native_snapshot_callback(case):
    pending = [False]

    def read(port):
        if port.scorer and case.work.get("returns"):
            pending[0] = True

    def parent(owner):
        if pending[0]:
            raise ValueError("preceding paid stream evidence changed")

    case.work.update(on_snapshot=read, on_guard=parent)
    with pytest.raises(evaluation.MiniGridEvaluationFailed) as caught:
        evaluate(case)
    assert caught.value.report["work"]["raw_score_forwards_completed"] == 1
    assert not case.work.get("environments")


def test_should_leave_the_original_raw_evaluator_closed_to_bound_snapshots(case):
    with pytest.raises(ValueError):
        evaluation.evaluate_checkpoint(
            case.source,
            case.state,
            method="observation_memory",
            updates=128,
            layouts=case.layouts,
            role="train",
            sources=case.sources,
            budget=case.budget,
            check_boundary=case.guard,
            public_history=True,
        )
    assert case.work == {}
