"""Fixed H32 methods use the ordinary isolated, paid TRAIN stream.

Per command: four two-pass routes (432 rows), four nine-row fork updates
(36 rows), and one lost-return BP update (54 rows): 13 calls / 522 rows.
Predictions: 48 catalogue calls / 432 rows plus eight / 72 fork rows.
No acquisition threshold, environment, teacher, probe, or final evaluation.
"""

from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import pickle
import random

import numpy as np
import pytest

from src.adapters import minigrid_constructive_fit as fit
from src.adapters.invariant_torch_feature_learner import (
    InvariantTorchFeaturePolicy,
    InvariantTorchFeatureSnapshot,
)
from src.adapters.torch_feature_learner import TorchFeatureLearner, TorchFeaturePolicy, TORCH_CE_ID
from src.core.demonstration_episode import episode_arrays
from src.core.resnet50_variants import TORCH_PC_ENERGY_ID
from src.shared.torch_runtime import require_torch
from test_minigrid_constructive_fit import assert_raw_scores, capture_stream
from test_minigrid_train_fit_bounds import histories, raw_sources
from test_torch_feature_learner import assert_tree, tensors
from test_torch_feature_rehearsal_pilot import forbidden


METHODS = ("bp", "pc", "neutral_cpc", "gated_cpc")


@pytest.fixture
def no_resources(monkeypatch):
    for name in ("_learner", "AttributionStream", "ProcessRssSampler"):
        monkeypatch.setattr(fit, name, forbidden)


@pytest.mark.parametrize("method", METHODS)
def test_should_describe_fixed_method_without_resources_or_changing_defaults(no_resources, method):
    assert fit.describe_protocol() == fit.describe_protocol(wide_method=None)
    default = fit.describe_protocol()
    assert default["hidden_dim"] == 8 and default["method"] == "public_invariant_bp"
    protocol = fit.describe_protocol(wide_method=method, updates=2, max_wall_seconds=10)
    assert protocol["definition"] == fit.WIDE_NEURAL_DEFINITION
    assert protocol["wide_neural_method"] == method
    assert protocol["method"] == f"public_invariant_{method}"
    assert protocol["hidden_dim"] == 32 and protocol["feature_dim"] == 296
    assert protocol["num_classes"] == 7 and protocol["seed"] == 17
    assert protocol["learning_rate"] == 0.03 and protocol["momentum"] == 0.9
    assert protocol["inference_steps"] == 2 and protocol["inference_learning_rate"] == 0.04
    assert protocol["parameter_elements"] == 296 * 32 + 32 + 32 * 7 + 7 == 9735
    assert protocol["parameter_array_equivalent_bytes"] == 38940
    assert protocol["native_diagnostic_definition"] == (
        TORCH_CE_ID if method == "bp" else TORCH_PC_ENERGY_ID
    )
    assert protocol["nominal_latent_row_steps"] == (0 if method == "bp" else 216)
    assert protocol["checkpoint_updates"] == [0, 2]
    assert protocol["training_rows"] == protocol["score_rows"] == 108
    assert protocol["replay_rows"] == protocol["probe_updates"] == protocol["final_episodes"] == 0
    assert protocol["heldout"] == [] and protocol["training_only"] is True


@pytest.mark.parametrize("method", [True, False, 1, 32, "BP", "", "unknown", np.str_("bp")])
def test_should_refuse_noncanonical_methods_before_resources(no_resources, method):
    with pytest.raises(ValueError):
        fit.describe_protocol(wide_method=method)
    with pytest.raises(ValueError):
        fit.diagnose_train_fit(histories(), wide_method=method, updates=2)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "other", ["perceptron", "kernel_perceptron", "cue_relative", "cue_turn_gradients"]
)
def test_should_refuse_mixed_families_before_resources(no_resources, method, other):
    options = dict(wide_method=method, **{other: True})
    with pytest.raises(ValueError):
        fit.describe_protocol(**options)
    with pytest.raises(ValueError):
        fit.diagnose_train_fit(histories(), updates=2, **options)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("fault", ["probe_train", "probe_test", "final_test", "raw148", "missing"])
def test_should_admit_only_complete_original_train_sources_before_resources(
    no_resources, method, fault
):
    sources = histories()
    if fault == "raw148":
        sources = raw_sources()
    elif fault == "missing":
        sources = sources[:-1]
    else:
        sources = (replace(sources[0], role=fault), *sources[1:])
    with pytest.raises(ValueError):
        fit.diagnose_train_fit(sources, wide_method=method, updates=2)


@pytest.mark.parametrize("method", METHODS)
def test_should_train_two_native_passes_and_score_only_detached_checkpoints(
    monkeypatch, tmp_path, method
):
    torch = require_torch()
    captured = capture_stream(monkeypatch)
    sources = histories()
    before = pickle.dumps(sources)
    rng = (random.getstate(), pickle.dumps(np.random.get_state()), torch.get_rng_state().clone())
    train = TorchFeatureLearner.train_batch
    diagnostics = []
    scoring = []
    original_score = fit._score_point

    def counted(owner, features, targets):
        result = train(owner, features, targets)
        diagnostics.append(result)
        assert len(diagnostics) <= 2 and len(features) == 54
        return result

    def score(source, state, point, episodes, report, guard):
        assert captured["stream"]._finished is True
        assert captured["stream"].completed_updates == 2
        fit._counter(state, point, wide_method=method)
        scoring.append(deepcopy(state))
        return original_score(source, state, point, episodes, report, guard)

    monkeypatch.setattr(TorchFeatureLearner, "train_batch", counted)
    monkeypatch.setattr(fit, "_score_point", score)
    result = fit.diagnose_train_fit(sources, wide_method=method, updates=2, max_wall_seconds=20)
    label = os.environ.get("WIDE_CHECK_LABEL")
    destination = (
        Path("artifacts/runs/wide-neural-train-fit-20261018") / f"{label}-{method}-synthetic.json"
        if label
        else tmp_path / f"{method}.json"
    )
    with destination.open("x", encoding="utf8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    assert result["status"] == "complete", result["reason"]
    assert (
        result["charged_updates"]
        == result["committed_updates"]
        == result["native_completed_updates"]
        == 2
    )
    assert result["native_training_rows"] == 108
    assert result["source_unchanged"] is result["actor_unchanged"] is True
    assert result["stream_result"]["evaluation_scope"] == "training_only"
    assert captured["heldout"] == () and captured["options"]["training_only"] is True
    assert captured["options"]["label_delay"] == 1
    assert len(captured["training"]) == 2
    assert all(batch.role == "train" for batch in captured["stream"]._training)
    assert captured["options"]["budget"].updates_completed == 2
    assert len(result["applied_updates"]) == 2
    assert [receipt["update_number"] for receipt in result["applied_updates"]] == [1, 2]
    assert all(
        receipt["arrived_at"] == receipt["observed_at"] + 1 for receipt in result["applied_updates"]
    )
    assert len(diagnostics) == 2
    assert all(
        d.definition == (TORCH_CE_ID if method == "bp" else TORCH_PC_ENERGY_ID) for d in diagnostics
    )
    assert [state.completed_updates for state in scoring] == [0, 2]
    assert all(
        state.policy.native_policy == TorchFeaturePolicy(method, 296, 32, 17, num_classes=7)
        for state in scoring
    )
    work = result["work"]
    assert work["score_forward_calls_attempted"] == work["score_forward_calls_completed"] == 12
    assert work["score_rows_completed"] == 108
    costs = work["wide_neural"]
    assert costs["completed_native_calls"] == 2
    assert costs["completed_training_rows"] == 108 and costs["method"] == method
    assert costs["hidden_dim"] == 32 and costs["parameter_elements"] == 9735
    assert costs["parameter_array_equivalent_bytes"] == 38940
    assert (
        costs["bp_backward_calls_derived"]
        == costs["bp_optimizer_steps_derived"]
        == (2 if method == "bp" else 0)
    )
    assert costs["nominal_latent_row_steps"] == (0 if method == "bp" else 216)
    assert costs["pc_parameter_updates_derived"] == (0 if method == "bp" else 2)
    assert "derived" in costs["scope"]
    assert work["training_work_may_be_unobserved"] is work["score_work_may_be_unobserved"] is False
    assert_raw_scores(result)
    assert_tree(torch, captured["source"].snapshot_state(), captured["initial"])
    assert_tree(
        torch, captured["stream"]._runtime.actor.snapshot_state().state, captured["initial"]
    )
    assert pickle.dumps(sources) == before
    assert random.getstate() == rng[0] and pickle.dumps(np.random.get_state()) == rng[1]
    assert torch.equal(torch.get_rng_state(), rng[2])


@pytest.mark.parametrize("method", METHODS)
def test_should_preserve_complete_owned_fork_and_refuse_forged_wide_checkpoints(method):
    torch = require_torch()
    source = fit._learner(wide_method=method)
    initial = source.snapshot_state()
    fit._counter(initial, 0, wide_method=method)
    child = source.fork()
    features, targets = episode_arrays(histories()[0])
    inputs = (features.copy(), targets.copy())
    child.train_batch(features, targets)
    trained = child.snapshot_state()
    assert trained.completed_updates == 1 and trained.training_examples == 9
    assert_tree(torch, source.snapshot_state(), initial)
    assert np.array_equal(features, inputs[0]) and np.array_equal(targets, inputs[1])
    source.restore_state(trained)
    assert_tree(torch, source.snapshot_state(), trained)
    caller_state = deepcopy(trained)
    source.restore_state(caller_state)
    tensors(torch, caller_state)[0].add_(1)
    assert_tree(torch, source.snapshot_state(), trained)
    assert np.array_equal(source.predict(features), child.predict(features))
    for left, right in zip(
        tensors(torch, source.snapshot_state()), tensors(torch, trained), strict=True
    ):
        assert left.data_ptr() != right.data_ptr()
    child.restore_state(initial)
    assert_tree(torch, child.snapshot_state(), initial)
    assert_tree(torch, source.snapshot_state(), trained)
    if method == "bp":
        assert trained.native_state.optimizer_state
        assert trained.native_state.gradients is not None
    else:
        assert trained.native_state.optimizer_state is None
    if method in ("neutral_cpc", "gated_cpc"):
        assert "split_generator_state" in trained.native_state.head_state
        assert "energy_history" in trained.native_state.head_state
    bad_states = []
    for field, value in (("completed_updates", True), ("training_examples", 1)):
        bad = deepcopy(initial)
        object.__setattr__(bad.native_state, field, value)
        bad_states.append(bad)
    rogue = deepcopy(initial)
    object.__setattr__(rogue, "foreign", True)
    bad_states.append(rogue)
    wrong_policy = InvariantTorchFeaturePolicy(
        TorchFeaturePolicy(method, 296, 8, 17, num_classes=7)
    )
    bad_states.append(InvariantTorchFeatureSnapshot(wrong_policy, deepcopy(initial.native_state)))
    bad_states.append(initial.native_state)
    for bad in bad_states:
        with pytest.raises(ValueError):
            fit._counter(bad, 0, wide_method=method)
    for point in (True, 1, -1, 0.0):
        with pytest.raises(ValueError):
            fit._counter(initial, point, wide_method=method)
    with pytest.raises(ValueError):
        fit._counter(initial, 0, wide_method="pc" if method == "bp" else "bp")
    assert_tree(torch, source.snapshot_state(), trained)


def test_should_preserve_native_prefix_when_a_completed_bp_return_is_lost(monkeypatch):
    train = TorchFeatureLearner.train_batch

    def lost(owner, features, targets):
        train(owner, features, targets)
        raise RuntimeError("injected lost completed H32 return")

    monkeypatch.setattr(TorchFeatureLearner, "train_batch", lost)
    result = fit.diagnose_train_fit(histories(), wide_method="bp", updates=2, max_wall_seconds=10)
    assert result["status"] == "incomplete" and "lost completed H32 return" in result["reason"]
    assert result["native_completed_updates"] == 1 and result["native_training_rows"] == 54
    assert result["charged_updates"] == result["committed_updates"] == 0
    assert result["applied_updates"] == [] and result["readouts"] == []
    assert result["work"]["training_work_may_be_unobserved"] is True
    assert result["work"]["wide_neural"]["completed_training_rows"] == 54
    assert result["work"]["wide_neural"]["bp_backward_calls_derived"] == 1
