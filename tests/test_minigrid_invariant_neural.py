"""Native neural TRAIN and frozen fake-game integration, not efficacy evidence."""

from copy import deepcopy
from time import monotonic
from typing import Any

import pytest

from src.adapters import minigrid_learning_pilot as training
from src.adapters import minigrid_policy_evaluation as evaluation
from src.adapters.action_memory_learner import ActionMemoryLearner
from src.adapters.invariant_torch_feature_learner import (
    InvariantTorchFeatureLearner,
    InvariantTorchFeaturePolicy,
)
from src.adapters.torch_feature_learner import TorchFeatureLearner, TorchFeaturePolicy
from src.adapters.torch_feature_rehearsal_pilot import _same_state
from src.app import toy_execution_budget as budgets
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.action_memory import ActionMemoryPolicy
from src.core.minigrid_invariant_encoding import INVARIANT_HISTORY_ENCODING_ID
from src.shared.torch_runtime import require_torch
from test_minigrid_invariant_evaluation import DiscreteEnvironment, Environment
from test_minigrid_invariant_training import training_packets
from test_torch_feature_rehearsal_pilot import Sampler, forbidden


METHODS = ("bp", "pc", "neutral_cpc", "gated_cpc")


def learner(method="pc", **options):
    return InvariantTorchFeatureLearner(
        InvariantTorchFeaturePolicy(
            TorchFeaturePolicy(method, 296, 8, 17, num_classes=7, **options)
        )
    )


@pytest.fixture
def case(monkeypatch):
    torch = require_torch()
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    monkeypatch.setattr(budgets, "ProcessRssSampler", Sampler)
    monkeypatch.setattr(evaluation, "MiniGridKeyDoor", DiscreteEnvironment)
    work: dict[str, Any] = {}
    monkeypatch.setattr(Environment, "work", work)
    return work


def readout(source, state=None, *, guard=lambda: None, budget=None, updates=0):
    if state is None:
        state = source.snapshot_state()
    if budget is None:
        budget = ToyBudgetSession(
            ToyExecutionBudget(max_training_updates=0, max_wall_seconds=30), monotonic
        )
    return evaluation.evaluate_neural_representation_checkpoint(
        source, state, updates=updates, budget=budget, check_boundary=guard
    )


@pytest.mark.parametrize("method", METHODS)
def test_should_run_existing_A32_stream_and_two_owned_target_free_readouts(case, method):
    row = training._row(f"public_invariant_{method}")
    packets = training_packets()
    arm = training._train_arm(
        row,
        packets,
        Sampler(),
        monotonic() + 60,
        lambda: None,
        learner_factory=lambda _: learner(method),
        checkpoint_updates=(0, 32),
    )
    assert row["charged_updates"] == 32 and row["native_training_examples"] == 288
    assert row["source_unchanged"] is row["actor_unchanged"] is True
    assert row["stream_result"]["evaluation_scope"] == "training_only"
    assert row["stream_result"]["score_forward_calls"] == 0
    assert row["stream_result"]["checkpoints"] == []
    receipts = row["partial_stream"]["applied_updates"]
    assert len(receipts) == 32 and row["partial_stream"]["committed_training_examples"] == 288
    assert row["wake_calls_attempted"] == row["wake_calls_returned"] == 32
    initial = arm["source"].snapshot_state()
    for point, state in arm["states"]:
        before = deepcopy(state)
        report = readout(arm["source"], state, updates=point, guard=arm["guard"])
        assert report["status"] == "complete"
        assert report["method"] == row["method"] and report["native_method"] == method
        assert report["model_encoding"] == INVARIANT_HISTORY_ENCODING_ID
        assert report["native_training_examples"] == 9 * point
        assert report["scores"] == [] and report["state_unchanged"] is True
        assert [(slot["context"], slot["source_role"]) for slot in report["episodes"]] == [
            ("A", "train"),
            ("A", "train"),
            ("E-blue", "probe_test"),
            ("E-red", "probe_test"),
        ]
        assert report["work"]["internal_projection_rows_returned"] == 8
        assert report["work"]["internal_projection_array_bytes_returned"] == 8 * 2368
        assert all(
            len(decision["features"]) == 296
            for slot in report["episodes"]
            for decision in slot["predictions"]
        )
        assert _same_state(state, before)
        arm["guard"]()
    assert _same_state(initial, arm["source"].snapshot_state())
    assert arm["budget"].updates_completed == 32 and arm["budget"].replay_examples_completed == 0


@pytest.mark.parametrize("fault", ["raw", "method", "cap", "encoding", "clock", "extra"])
def test_should_refuse_wrong_training_factory_before_any_update(case, monkeypatch, fault):
    source: Any = learner("pc", **({"learning_rate": 0.04} if fault == "cap" else {}))
    if fault == "raw":
        source = TorchFeatureLearner(TorchFeaturePolicy("pc", 296, 8, 17, num_classes=7))
    if fault == "method":
        source = learner("bp")
    if fault in ("encoding", "clock", "extra"):
        state = source.snapshot_state()
        if fault == "encoding":
            object.__setattr__(state.policy, "encoding_id", "raw")
        elif fault == "clock":
            object.__setattr__(state.native_state, "completed_updates", True)
        else:
            object.__setattr__(state, "extra", None)
        monkeypatch.setattr(source, "snapshot_state", lambda: state)
    monkeypatch.setattr(source, "train_batch", forbidden)
    row = training._row("public_invariant_pc")
    with pytest.raises(ValueError):
        training._train_arm(
            row,
            training_packets(),
            Sampler(),
            monotonic() + 30,
            lambda: None,
            learner_factory=lambda _: source,
            checkpoint_updates=(0, 32),
        )
    assert row["charged_updates"] is None and row["wake_calls_attempted"] == 0


@pytest.mark.parametrize("fault", ["raw", "count", "clock", "policy", "budget", "label"])
def test_should_refuse_invalid_readout_before_any_fork_or_environment(case, monkeypatch, fault):
    source: Any = learner()
    state: Any = source.snapshot_state()
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=0), monotonic)
    point = 0
    if fault == "raw":
        state = state.native_state
    elif fault == "count":
        state = ActionMemoryLearner(
            ActionMemoryPolicy(feature_dim=296, num_classes=7, max_payload_bytes=155136)
        ).snapshot_state()
    elif fault == "clock":
        object.__setattr__(state.native_state, "training_examples", 1)
    elif fault == "policy":
        state = learner(learning_rate=0.04).snapshot_state()
    elif fault == "budget":
        budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=1), monotonic)
    else:
        point = True
    monkeypatch.setattr(source, "fork", forbidden)
    with pytest.raises(ValueError):
        readout(source, state, budget=budget, updates=point)
    assert not case.get("environments")


def test_should_keep_original_count_and_neural_admission_separate(case, monkeypatch):
    source = learner()
    monkeypatch.setattr(source, "fork", forbidden)
    with pytest.raises(ValueError):
        evaluation.evaluate_representation_checkpoint(
            source,
            source.snapshot_state(),
            updates=0,
            budget=ToyBudgetSession(ToyExecutionBudget(max_training_updates=0), monotonic),
            check_boundary=lambda: None,
        )


def test_should_preserve_paid_prediction_when_native_state_changes_same_clock(case, monkeypatch):
    original = InvariantTorchFeatureLearner.predict

    def corrupt(owner, features):
        result = original(owner, features)
        state = owner.snapshot_state()
        state.native_state.head_state["bias_hidden"][0, 0] += 0.1
        owner.restore_state(state)
        return result

    monkeypatch.setattr(InvariantTorchFeatureLearner, "predict", corrupt)
    with pytest.raises(evaluation.MiniGridEvaluationFailed) as rejected:
        readout(learner())
    report = rejected.value.report
    assert report["status"] == "failed" and len(report["episodes"]) == 4
    assert report["work"]["live_prediction_calls_returned"] == 1
    assert report["work"]["environment_steps_returned"] == 0
    assert report["work"]["internal_projection_rows_returned"] == 1
    assert all(slot["status"] == "not_run" for slot in report["episodes"][1:])


def test_should_validate_original_training_guard_before_any_neural_work(case, monkeypatch):
    source = learner()
    monkeypatch.setattr(source, "fork", forbidden)

    def changed():
        raise ValueError("preceding success corrupted")

    with pytest.raises(evaluation.MiniGridEvaluationFailed, match="preceding success") as rejected:
        readout(source, guard=changed)
    assert rejected.value.report["work"]["live_prediction_calls_attempted"] == 0
    assert all(slot["status"] == "not_run" for slot in rejected.value.report["episodes"])
