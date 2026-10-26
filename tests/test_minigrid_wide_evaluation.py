"""H32 admission and owned frozen forwards on controlled fake game frames.

The endpoint fixture has a structurally valid128/6912 native clock, created
without training. Its weights and adaptive values are engineering inputs, not
an executed128-pass checkpoint or evidence of acquisition or game success.
Per command: zero native updates,22 native prediction calls/rows, zero actual
physics. The two-step fake performs16 returned steps across successful slots
and failure checks; it supplies no teacher, heldout or final observations.
"""

from copy import deepcopy
from time import monotonic
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.adapters import minigrid_policy_evaluation as evaluation
from src.adapters.invariant_torch_feature_learner import (
    InvariantTorchFeatureLearner,
    InvariantTorchFeaturePolicy,
    copy_invariant_torch_feature_snapshot,
)
from src.adapters.torch_feature_learner import TorchFeatureLearner, TorchFeaturePolicy
from src.adapters.torch_feature_rehearsal_pilot import _same_state
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.minigrid_invariant_encoding import INVARIANT_HISTORY_ENCODING_ID
from src.shared.torch_runtime import require_torch
from test_minigrid_invariant_evaluation import DiscreteEnvironment, Environment
from test_torch_feature_rehearsal_pilot import forbidden


METHODS = ("bp", "pc", "neutral_cpc", "gated_cpc")


def fixed_policy(method="bp", **options):
    values = dict(method=method, feature_dim=296, hidden_dim=32, seed=17, num_classes=7)
    values.update(options)
    return InvariantTorchFeaturePolicy(TorchFeaturePolicy(**values))


def declared_endpoint(source):
    """Build a valid-clock test input; never imply it was trained."""
    torch = require_torch()
    state = source.snapshot_state()
    native = state.native_state
    method = native.policy.method
    object.__setattr__(native, "completed_updates", 128)
    object.__setattr__(native, "training_examples", 6912)
    if method == "bp":
        names = ("weight_feature_hidden", "bias_hidden", "weight_hidden_output", "bias_output")
        native.optimizer_state["state"].update(
            {
                index: {"momentum_buffer": torch.zeros_like(native.head_state[name])}
                for index, name in enumerate(names)
            }
        )
        native.gradients.update({name: torch.zeros_like(native.head_state[name]) for name in names})
    elif method == "pc":
        native.head_state["traffic_steps"] = 128
    else:
        head = native.head_state
        head.update(traffic_steps=128, wake_examples=6912, steps_since_sleep=128)
        head["neuron_age"].fill_(128)
        history_limit = max(head["config"]["sleep_energy_window"] * 4, 16)
        head["energy_history"] = [0.0] * min(128, history_limit)
    return copy_invariant_torch_feature_snapshot(state)


@pytest.fixture
def case(monkeypatch):
    torch = require_torch()
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    work: dict[str, Any] = {}
    monkeypatch.setattr(Environment, "work", work)
    monkeypatch.setattr(evaluation, "MiniGridKeyDoor", DiscreteEnvironment)
    monkeypatch.setattr(TorchFeatureLearner, "train_batch", forbidden)
    monkeypatch.setattr(InvariantTorchFeatureLearner, "train_batch", forbidden)
    original_fork = InvariantTorchFeatureLearner.fork
    original_predict = InvariantTorchFeatureLearner.predict

    def fork(owner):
        work.setdefault("forks", []).append(owner)
        if work.get("identity_fork"):
            return owner
        child = original_fork(owner)
        work.setdefault("scorers", []).append(child)
        return child

    def predict(owner, features):
        work.setdefault("attempts", []).append((owner, features.copy()))
        if work.get("prediction_error"):
            raise RuntimeError("injected wide prediction failure")
        result = original_predict(owner, features)
        work.setdefault("predictions", []).append((features.copy(), result.copy()))
        callback = work.get("on_predict")
        if callback is not None:
            callback(owner)
        if work.get("bad_probability"):
            return np.ones((1, 6), dtype=np.float64)
        return result

    monkeypatch.setattr(InvariantTorchFeatureLearner, "fork", fork)
    monkeypatch.setattr(InvariantTorchFeatureLearner, "predict", predict)

    def build(method="bp", **policy_options):
        source = InvariantTorchFeatureLearner(fixed_policy(method, **policy_options))
        budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=0), monotonic)
        binding = (budget.budget, budget.clock, budget.started_at, budget.progress)

        def guard():
            work.setdefault("guards", []).append(len(work.get("predictions", [])))

        return SimpleNamespace(
            source=source,
            state=declared_endpoint(source),
            method=method,
            budget=budget,
            binding=binding,
            guard=guard,
            work=work,
        )

    return build


def evaluate(current, **options):
    arguments = dict(
        updates=128,
        budget=current.budget,
        check_boundary=current.guard,
        known_a_only=True,
        wide_method=current.method,
    )
    arguments.update(options)
    return evaluation.evaluate_neural_representation_checkpoint(
        current.source, current.state, **arguments
    )


@pytest.mark.parametrize("method", METHODS)
def test_should_run_only_both_train_a_cues_on_an_owned_frozen_wide_reader(case, method):
    current = case(method)
    original = deepcopy(current.state)
    source_before = current.source.snapshot_state()
    torch = require_torch()
    rng = torch.get_rng_state().clone()
    report = evaluate(current)
    assert report["status"] == "complete" and report["state_unchanged"] is True
    assert report["method"] == f"public_invariant_{method}"
    assert report["native_method"] == report["wide_neural_method"] == method
    assert report["hidden_dim"] == 32 and report["catalogue_rows_per_update"] == 54
    assert report["native_training_examples"] == 6912 and report["updates"] == 128
    assert report["model_encoding"] == INVARIANT_HISTORY_ENCODING_ID
    assert report["scores"] == [] and report["role"] == "knownTRAIN_A_only"
    assert [(slot["context"], slot["cue"], slot["source_role"]) for slot in report["episodes"]] == [
        ("A", "blue", "train"),
        ("A", "red", "train"),
    ]
    assert len(current.work["forks"]) == len(current.work["scorers"]) == 1
    scorer = current.work["scorers"][0]
    assert scorer is not current.source
    assert _same_state(scorer.snapshot_state(), original)
    assert _same_state(current.source.snapshot_state(), source_before)
    assert _same_state(current.state, original)
    assert torch.equal(torch.get_rng_state(), rng)
    assert (
        current.budget.budget,
        current.budget.clock,
        current.budget.started_at,
        current.budget.progress,
    ) == current.binding
    assert current.budget.updates_completed == current.budget.replay_examples_completed == 0
    for slot, environment in zip(report["episodes"], current.work["environments"], strict=True):
        assert environment.closed and environment.seed == 17 and environment.max_steps == 16
        assert len(slot["predictions"]) == 2
        first = environment.transitions[0]["observation"]
        for decision, transition in zip(slot["predictions"], environment.transitions, strict=True):
            assert tuple(decision["features"]) == first + transition["observation"]
            probabilities = np.array(decision["probabilities"])
            assert probabilities.shape == (7,) and np.isclose(probabilities.sum(), 1)
            assert decision["action"] == transition["action"] == int(np.argmax(probabilities))
    work = report["work"]
    assert work["raw_score_forwards_attempted"] == work["raw_score_rows_completed"] == 0
    assert work["live_prediction_calls_returned"] == work["environment_steps_returned"] == 4
    assert work["environment_resets_completed"] == 2
    assert work["internal_projection_rows_returned"] == 4
    assert work["internal_projection_array_bytes_returned"] == 4 * 296 * 8


@pytest.mark.parametrize("method", [True, False, 1, "", "BP", "other", np.str_("bp")])
def test_should_refuse_noncanonical_wide_method_before_callbacks(case, method):
    current = case()
    with pytest.raises(ValueError):
        evaluate(current, wide_method=method)
    assert not current.work.get("guards") and not current.work.get("forks")
    assert not current.work.get("environments")


@pytest.mark.parametrize(
    "options",
    [
        dict(known_a_only=False),
        dict(known_a_only=1),
        dict(updates=True),
        dict(updates=0),
        dict(updates=64),
        dict(catalogue_method="public_invariant_bp"),
        dict(check_boundary=None),
    ],
)
def test_should_refuse_wrong_wide_route_or_label_before_callbacks(case, options):
    current = case()
    with pytest.raises(ValueError):
        evaluate(current, **options)
    assert not current.work.get("guards") and not current.work.get("forks")
    assert not current.work.get("environments")


@pytest.mark.parametrize(
    "policy_options",
    [
        dict(hidden_dim=8),
        dict(seed=18),
        dict(learning_rate=0.04),
        dict(inference_steps=3),
        dict(inference_learning_rate=0.05),
        dict(momentum=0.8),
    ],
)
def test_should_refuse_foreign_native_recipe_before_callbacks(case, policy_options):
    current = case(**policy_options)
    with pytest.raises(ValueError):
        evaluate(current)
    assert not current.work.get("guards") and not current.work.get("forks")
    assert not current.work.get("environments")


@pytest.mark.parametrize(
    "fault",
    [
        "family",
        "encoding",
        "method",
        "bool_updates",
        "bool_rows",
        "rows",
        "clock",
        "rogue",
        "optimizer",
    ],
)
def test_should_validate_complete_checkpoint_before_callbacks(case, fault):
    current = case()
    if fault == "family":
        current.state = current.state.native_state
    elif fault == "encoding":
        object.__setattr__(current.state.policy, "encoding_id", "foreign")
    elif fault == "method":
        current.method = "pc"
    elif fault in ("bool_updates", "clock"):
        object.__setattr__(
            current.state.native_state,
            "completed_updates",
            True if fault == "bool_updates" else 127,
        )
    elif fault in ("bool_rows", "rows"):
        object.__setattr__(
            current.state.native_state, "training_examples", True if fault == "bool_rows" else 6911
        )
    elif fault == "rogue":
        object.__setattr__(current.state, "foreign", 1)
    else:
        current.state.native_state.optimizer_state["state"].clear()
    with pytest.raises(ValueError):
        evaluate(current)
    assert not current.work.get("guards") and not current.work.get("forks")
    assert not current.work.get("environments")


@pytest.mark.parametrize("fault", ["rng", "adaptive"])
def test_should_validate_complete_cpc_rng_and_adaptive_state_before_callbacks(case, fault):
    current = case("gated_cpc")
    head = current.state.native_state.head_state
    if fault == "rng":
        head["split_generator_state"] = None
    else:
        head["wake_examples"] = 6911
    with pytest.raises(ValueError):
        evaluate(current)
    assert not current.work.get("guards") and not current.work.get("forks")
    assert not current.work.get("environments")


@pytest.mark.parametrize(
    "fault", ["source_endpoint", "source_clock", "identity_fork", "budget", "guard"]
)
def test_should_refuse_changed_source_or_read_budget_before_predictions(case, fault):
    current = case()
    if fault == "source_endpoint":
        current.source.restore_state(current.state)
    elif fault == "source_clock":
        original = current.source.snapshot_state

        def malformed():
            state = original()
            object.__setattr__(state.native_state, "completed_updates", True)
            return state

        current.source.snapshot_state = malformed
    elif fault == "identity_fork":
        current.work["identity_fork"] = True
    elif fault == "budget":
        current.budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=1), monotonic)
    else:

        def refuse():
            raise ValueError("preceding TRAIN evidence replaced")

        current.guard = refuse
    with pytest.raises((ValueError, evaluation.MiniGridEvaluationFailed)):
        evaluate(current)
    assert not current.work.get("attempts") and not current.work.get("environments")


@pytest.mark.parametrize("target", ["source", "scorer", "checkpoint", "budget"])
def test_should_refuse_same_counter_payload_or_budget_changes_after_paid_forward(case, target):
    current = case()

    def mutate(owner):
        if target == "checkpoint":
            current.state.native_state.head_state["bias_hidden"][0, 0] += 0.25
        elif target == "budget":
            current.budget.clock = lambda: current.budget.started_at
        else:
            selected = current.source if target == "source" else owner
            state = selected.snapshot_state()
            state.native_state.head_state["bias_hidden"][0, 0] += 0.25
            selected.restore_state(state)

    current.work["on_predict"] = mutate
    with pytest.raises(evaluation.MiniGridEvaluationFailed) as rejected:
        evaluate(current)
    report = rejected.value.report
    assert report["work"]["live_prediction_calls_returned"] == 1
    assert report["work"]["environment_steps_attempted"] == 0
    assert [slot["status"] for slot in report["episodes"]] == ["failed", "not_run"]
    assert current.work["environments"][0].closed


@pytest.mark.parametrize("fault", ["prediction_error", "bad_probability", "step_error"])
def test_should_retain_paid_live_failure_prefix_and_second_unreached_cue(case, fault):
    current = case()
    current.work[fault] = True
    with pytest.raises(evaluation.MiniGridEvaluationFailed) as rejected:
        evaluate(current)
    report = rejected.value.report
    assert report["status"] == "failed" and report["scores"] == []
    assert [slot["status"] for slot in report["episodes"]] == ["failed", "not_run"]
    work = report["work"]
    assert work["live_prediction_calls_attempted"] == 1
    assert work["live_prediction_calls_returned"] == int(fault != "prediction_error")
    assert work["environment_steps_attempted"] == int(fault == "step_error")
    assert work["environment_steps_returned"] == 0
    assert work["failure_work_may_be_unobserved"] is True
    assert current.work["environments"][0].closed
