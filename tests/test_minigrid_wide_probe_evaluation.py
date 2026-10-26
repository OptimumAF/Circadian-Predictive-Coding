"""Fixed H32 disposable endpoints on controlled fake frames, never training.

Clock fixtures explicitly fabricate valid optimizer/adaptive counters; they are
not executed training or acquisition evidence. Success cases make one real
native prediction per method/origin/phase combination, then scripted uniform
readouts:16 native calls/rows total, zero updates and zero actual physics.
Other tests use scripted probability ports and real owned snapshot validation.
"""

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.adapters import minigrid_policy_evaluation as evaluation
from src.adapters.invariant_torch_feature_learner import (
    InvariantTorchFeatureLearner,
    copy_invariant_torch_feature_snapshot,
)
from src.adapters.torch_feature_learner import TorchFeatureLearner
from src.adapters.torch_feature_rehearsal_pilot import _same_state
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.shared.torch_runtime import require_torch
from test_minigrid_invariant_evaluation import DiscreteEnvironment, Environment
from test_minigrid_wide_evaluation import METHODS, fixed_policy
from test_torch_feature_rehearsal_pilot import forbidden


def declared_endpoint(source, origin, phase):
    """Engineering-only full state with literal mixed clock law, no training."""
    torch = require_torch()
    state = source.snapshot_state()
    native = state.native_state
    updates = origin + (16 if phase == "after_probe" else 32)
    rows = 54 * origin + 9 * (updates - origin)
    object.__setattr__(native, "completed_updates", updates)
    object.__setattr__(native, "training_examples", rows)
    method = native.policy.method
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
        native.head_state["traffic_steps"] = updates
    else:
        head = native.head_state
        head.update(traffic_steps=updates, wake_examples=rows, steps_since_sleep=updates)
        head["neuron_age"].fill_(updates)
        limit = max(head["config"]["sleep_energy_window"] * 4, 16)
        head["energy_history"] = [0.0] * min(updates, limit)
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
            raise RuntimeError("injected H32 probe prediction failure")
        if work.get("native_once") and not work.get("native_returns"):
            result = original_predict(owner, features)
            work["native_returns"] = 1
        else:
            result = np.full((1, 7), 1 / 7, dtype=np.float64)
        work.setdefault("predictions", []).append((features.copy(), result.copy()))
        callback = work.get("on_predict")
        if callback:
            callback(owner)
        if work.get("bad_probability"):
            return np.ones((1, 6), dtype=np.float64)
        return result

    monkeypatch.setattr(InvariantTorchFeatureLearner, "fork", fork)
    monkeypatch.setattr(InvariantTorchFeatureLearner, "predict", predict)

    def build(method="bp", origin=0, phase="after_probe", **policy_options):
        source = InvariantTorchFeatureLearner(fixed_policy(method, **policy_options))
        state = declared_endpoint(source, origin, phase)
        source.restore_state(state)
        budget = ToyBudgetSession(
            ToyExecutionBudget(max_training_updates=0, max_replay_examples=0), lambda: 10.0
        )
        current = SimpleNamespace(
            source=source,
            state=state,
            method=method,
            origin=origin,
            phase=phase,
            budget=budget,
            work=work,
        )

        def guard():
            work["guards"] = work.get("guards", 0) + 1
            callback = work.get("on_guard")
            if callback:
                callback(current)

        current.guard = guard
        return current

    return build


def evaluate(current, **options):
    values = dict(
        source_updates=current.origin,
        phase=current.phase,
        method=current.method,
        budget=current.budget,
        check_boundary=current.guard,
    )
    values.update(options)
    return evaluation.evaluate_wide_neural_probe_checkpoint(current.source, current.state, **values)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("origin", [0, 128])
@pytest.mark.parametrize("phase", ["after_probe", "after_return"])
def test_should_freeze_each_native_endpoint_and_run_only_fixed_a_f_cues(
    case, method, origin, phase
):
    current = case(method, origin, phase)
    current.work["native_once"] = True
    before = deepcopy(current.state)
    source_before = current.source.snapshot_state()
    torch = require_torch()
    rng = torch.get_rng_state().clone()
    binding = (
        current.budget.budget,
        current.budget.clock,
        current.budget.started_at,
        current.budget.progress,
        current.budget.process_rss_sampler,
    )
    report = evaluate(current)
    assert report["status"] == "complete" and report["state_unchanged"] is True
    relative = 16 if phase == "after_probe" else 32
    assert (report["source_updates"], report["relative_updates"], report["phase"]) == (
        origin,
        relative,
        phase,
    )
    assert report["updates"] == origin + relative
    assert report["native_training_examples"] == 54 * origin + 9 * relative
    assert report["native_method"] == report["wide_neural_method"] == method
    assert report["hidden_dim"] == 32 and report["probe_rows_per_update"] == 9
    assert report["catalogue_rows_per_update"] == 54
    assert report["scores"] == [] and report["role"] == "same_fork_probe_exploratory"
    assert [(v["context"], v["cue"], v["source_role"]) for v in report["episodes"]] == [
        ("A", "blue", "train"),
        ("A", "red", "train"),
        ("F-blue", "blue", "probe_test"),
        ("F-red", "red", "probe_test"),
    ]
    assert len(current.work["forks"]) == len(current.work["scorers"]) == 1
    assert current.work["native_returns"] == 1
    assert _same_state(current.state, before)
    assert _same_state(current.source.snapshot_state(), source_before)
    assert _same_state(current.work["scorers"][0].snapshot_state(), before)
    assert torch.equal(torch.get_rng_state(), rng)
    assert (
        current.budget.budget,
        current.budget.clock,
        current.budget.started_at,
        current.budget.progress,
        current.budget.process_rss_sampler,
    ) == binding
    assert current.budget.updates_completed == current.budget.replay_examples_completed == 0
    for slot, env in zip(report["episodes"], current.work["environments"], strict=True):
        assert env.closed and env.seed == 17 and env.max_steps == 16
        assert slot["status"] == "complete" and len(slot["predictions"]) == 2
        first = slot["predictions"][0]["observation"]
        for decision, transition in zip(slot["predictions"], env.transitions, strict=True):
            assert decision["features"] == first + decision["observation"]
            assert (
                decision["action"]
                == transition["action"]
                == int(np.argmax(decision["probabilities"]))
            )
    work = report["work"]
    assert work["raw_score_forwards_attempted"] == work["raw_score_rows_completed"] == 0
    assert work["live_prediction_calls_returned"] == work["environment_steps_returned"] == 8
    assert work["environment_resets_completed"] == 4
    assert work["internal_projection_rows_returned"] == 8
    assert work["internal_projection_array_bytes_returned"] == 8 * 296 * 8


@pytest.mark.parametrize(
    "options",
    [
        dict(source_updates=True),
        dict(source_updates=np.int64(0)),
        dict(source_updates=32),
        dict(phase="before_probe"),
        dict(phase=np.str_("after_probe")),
        dict(phase=None),
        dict(method=True),
        dict(method=np.str_("bp")),
        dict(method="BP"),
        dict(method="other"),
        dict(check_boundary=None),
    ],
)
def test_should_refuse_foreign_options_before_snapshot_copy_or_callbacks(
    case, monkeypatch, options
):
    current = case()
    monkeypatch.setattr(evaluation, "copy_invariant_torch_feature_snapshot", forbidden)
    with pytest.raises(ValueError):
        evaluate(current, **options)
    assert current.work == {}


@pytest.mark.parametrize(
    "policy_options",
    [
        dict(hidden_dim=8),
        dict(seed=18),
        dict(learning_rate=0.04),
        dict(momentum=0.8),
        dict(inference_steps=3),
        dict(inference_learning_rate=0.05),
    ],
)
def test_should_refuse_foreign_native_recipe_before_resources(case, policy_options):
    current = case(**policy_options)
    with pytest.raises(ValueError):
        evaluate(current)
    assert current.work == {}


@pytest.mark.parametrize("origin", [0, 128])
@pytest.mark.parametrize("phase", ["after_probe", "after_return"])
@pytest.mark.parametrize("fault", ["clock", "rows", "old_nine_row_law", "bool_clock", "bool_rows"])
def test_should_refuse_wrong_mixed_native_clocks_before_resources(case, origin, phase, fault):
    current = case(origin=origin, phase=phase)
    native = current.state.native_state
    if fault == "clock":
        object.__setattr__(native, "completed_updates", native.completed_updates + 1)
    elif fault == "bool_clock":
        object.__setattr__(native, "completed_updates", True)
    else:
        rows = (
            True
            if fault == "bool_rows"
            else 9 * native.completed_updates
            if fault == "old_nine_row_law"
            else native.training_examples + 1
        )
        if origin == 0 and fault == "old_nine_row_law":
            rows += 1  # At origin0 the laws coincide; corrupt instead of claiming refusal.
        object.__setattr__(native, "training_examples", rows)
    with pytest.raises(ValueError):
        evaluate(current)
    assert current.work == {}


@pytest.mark.parametrize(
    "fault", ["family", "encoding", "method", "rogue", "optimizer", "gradients"]
)
def test_should_validate_the_full_owned_checkpoint_before_callbacks(case, fault):
    current = case()
    if fault == "family":
        current.state = current.state.native_state
    elif fault == "encoding":
        object.__setattr__(current.state.policy, "encoding_id", "foreign")
    elif fault == "method":
        current.method = "pc"
    elif fault == "rogue":
        object.__setattr__(current.state, "extra", 1)
    elif fault == "optimizer":
        current.state.native_state.optimizer_state["state"].clear()
    else:
        current.state.native_state.gradients.clear()
    with pytest.raises(ValueError):
        evaluate(current)
    assert current.work == {}


@pytest.mark.parametrize("fault", ["rng", "adaptive"])
def test_should_refuse_corrupted_cpc_rng_or_adaptive_state(case, fault):
    current = case("gated_cpc", 128)
    head = current.state.native_state.head_state
    if fault == "rng":
        head["split_generator_state"] = None
    else:
        head["wake_examples"] -= 1
    with pytest.raises(ValueError):
        evaluate(current)
    assert current.work == {}


@pytest.mark.parametrize(
    "fault", ["source_initial", "source_clock", "source_payload", "identity_fork", "guard"]
)
def test_should_require_the_actual_full_endpoint_source_before_predictions(case, fault):
    current = case()
    if fault in ("source_initial", "source_clock", "source_payload"):
        value = current.source.snapshot_state()
        if fault == "source_initial":
            fresh = InvariantTorchFeatureLearner(fixed_policy()).snapshot_state()
            current.source.restore_state(fresh)
        elif fault == "source_clock":
            replacement = declared_endpoint(current.source, 0, "after_return")
            current.source.restore_state(replacement)
        else:
            value.native_state.head_state["bias_hidden"][0, 0] += 0.25
            current.source.restore_state(value)
    elif fault == "identity_fork":
        current.work["identity_fork"] = True
    else:

        def reject():
            raise ValueError("preceding paid probe replaced")

        current.guard = reject
    with pytest.raises(evaluation.MiniGridEvaluationFailed):
        evaluate(current)
    assert not current.work.get("attempts") and not current.work.get("environments")
    if fault != "identity_fork":
        assert not current.work.get("forks")


@pytest.mark.parametrize("training,replay", [(1, 0), (0, None), (0, 1), (0, True)])
def test_should_require_no_update_or_replay_authority_before_callbacks(case, training, replay):
    current = case()
    object.__setattr__(current.budget.budget, "max_training_updates", training)
    object.__setattr__(current.budget.budget, "max_replay_examples", replay)
    with pytest.raises(ValueError):
        evaluate(current)
    assert current.work == {}


@pytest.mark.parametrize("target", ["source", "scorer", "checkpoint", "budget", "rng", "gradient"])
def test_should_refuse_same_counter_mutations_after_paid_prediction_before_physics(case, target):
    current = case("gated_cpc" if target == "rng" else "bp")

    def mutate(owner):
        if target == "budget":
            current.budget.clock = lambda: 10.0
        elif target == "checkpoint":
            current.state.native_state.head_state["bias_hidden"][0, 0] += 0.25
        elif target == "gradient":
            current.state.native_state.gradients["bias_hidden"][0, 0] += 0.25
        else:
            selected = current.source if target == "source" else owner
            value = selected.snapshot_state()
            if target == "rng":
                value.native_state.head_state["split_generator_state"] = (
                    require_torch().get_rng_state().clone()
                )
                value.native_state.head_state["split_generator_state"][0] ^= 1
            else:
                value.native_state.head_state["bias_hidden"][0, 0] += 0.25
            selected.restore_state(value)

    current.work["on_predict"] = mutate
    with pytest.raises(evaluation.MiniGridEvaluationFailed) as rejected:
        evaluate(current)
    report = rejected.value.report
    assert report["work"]["live_prediction_calls_returned"] == 1
    assert report["work"]["environment_steps_returned"] == 0
    assert [v["status"] for v in report["episodes"]] == ["failed", "not_run", "not_run", "not_run"]
    assert current.work["environments"][0].closed


@pytest.mark.parametrize(
    "fault", ["prediction_error", "bad_probability", "step_error", "close_error"]
)
def test_should_retain_paid_failure_and_all_unreached_slots(case, fault):
    current = case()
    current.work[fault] = True
    with pytest.raises(evaluation.MiniGridEvaluationFailed) as rejected:
        evaluate(current)
    report = rejected.value.report
    work = report["work"]
    assert work["failure_work_may_be_unobserved"] is True
    assert [v["status"] for v in report["episodes"]] == ["failed", "not_run", "not_run", "not_run"]
    assert work["live_prediction_calls_attempted"] == (2 if fault == "close_error" else 1)
    assert work["live_prediction_calls_returned"] == (
        0 if fault == "prediction_error" else 2 if fault == "close_error" else 1
    )
    assert work["environment_steps_returned"] == (2 if fault == "close_error" else 0)
    assert current.work["environments"][0].closed


def test_should_recheck_parent_after_final_snapshot_callback(case, monkeypatch):
    current = case()
    original = InvariantTorchFeatureLearner.snapshot_state
    changed = [False]

    def late(owner):
        result = original(owner)
        if owner in current.work.get("scorers", []) and current.work.get("predictions"):
            changed[0] = True
        return result

    def guard(owner):
        if changed[0]:
            raise ValueError("preceding paid report replaced in snapshot callback")

    monkeypatch.setattr(InvariantTorchFeatureLearner, "snapshot_state", late)
    current.work["on_guard"] = guard
    with pytest.raises(evaluation.MiniGridEvaluationFailed) as rejected:
        evaluate(current)
    assert rejected.value.report["work"]["live_prediction_calls_returned"] == 1
    assert rejected.value.report["work"]["environment_steps_returned"] == 0


def test_should_refuse_corrupted_public_input_before_any_prediction(case):
    current = case()

    def corrupt(owner):
        environments = current.work.get("environments", [])
        if environments:
            environments[0].observation = (float("nan"),) + (-1.0,) * 147

    current.work["on_guard"] = corrupt
    with pytest.raises(evaluation.MiniGridEvaluationFailed) as rejected:
        evaluate(current)
    assert rejected.value.report["work"]["live_prediction_calls_attempted"] == 0
    assert rejected.value.report["work"]["environment_steps_returned"] == 0
    assert current.work["environments"][0].closed


def test_should_refuse_source_change_during_boundary_before_new_prediction(case):
    current = case()
    changed = [False]

    def mutate(owner):
        if current.work.get("environments") and not changed[0]:
            state = current.source.snapshot_state()
            state.native_state.head_state["bias_hidden"][0, 0] += 0.25
            current.source.restore_state(state)
            changed[0] = True

    current.work["on_guard"] = mutate
    with pytest.raises(evaluation.MiniGridEvaluationFailed) as rejected:
        evaluate(current)
    assert rejected.value.report["work"]["live_prediction_calls_attempted"] == 0
    assert rejected.value.report["work"]["environment_steps_returned"] == 0
    assert current.work["environments"][0].closed
