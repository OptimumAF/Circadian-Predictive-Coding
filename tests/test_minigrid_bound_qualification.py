"""Existing bound learner selection with fixed qualification dose and fake IO.

Native factory cases construct only zero-update states. Composition reuses the
accepted TRAIN stub and real fit/probe guards with fake native learner ports.
No neural updates, SVDs, real environments or scientific comparisons are executed.
"""

from dataclasses import asdict
import json
import pickle
from typing import Any

import numpy as np
import pytest

from src.adapters import minigrid_learnability as pilot
from src.adapters.invariant_torch_feature_learner import InvariantTorchFeatureLearner
from src.adapters.torch_feature_learner import TorchFeatureLearner, TorchFeaturePolicy, _config
from src.core.minigrid_invariant_encoding import INVARIANT_HISTORY_ENCODING_ID
from src.shared.torch_runtime import require_torch
from test_minigrid_learning_pilot import POINTS, evaluator as evaluator
from test_minigrid_pc_qualification import environment
from test_torch_feature_rehearsal_pilot import forbidden


METHODS = ("bp", "pc", "neutral_cpc", "gated_cpc")
HISTORY: dict[str, Any] = dict(public_history=True, invariant_history=True)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("health", [False, True])
def test_should_describe_only_the_selected_original_family_with_frozen_dose_and_health_costs(
    monkeypatch, method, health
):
    for name in ("require_torch", "ProcessRssSampler", "_history_learner", "MiniGridKeyDoor"):
        monkeypatch.setattr(pilot, name, forbidden)
    protocol = pilot.describe_protocol(
        **HISTORY, qualification_method=method, activation_health=health
    )
    selected = f"public_invariant_{method}"
    suffix = "_native_activation_health" if health else ""
    assert (
        protocol["protocol_id"]
        == f"minigrid_invariant_{method}_learnability_qualification{suffix}_v1"
    )
    assert protocol["methods"] == [selected] and protocol["primary_comparisons"] == []
    assert protocol["feature_dim_by_method"] == {selected: 296}
    assert protocol["trainable_parameters"] == {selected: 2439}
    assert protocol["public_history"]["parameters_by_method"] == {selected: 2439}
    expected_configs = (
        {method: asdict(_config(TorchFeaturePolicy(method, 296, 8, 17, num_classes=7)))}
        if method in ("neutral_cpc", "gated_cpc")
        else {}
    )
    assert protocol["cpc_configs"] == expected_configs
    assert "recurrent_bc" not in protocol["invariant_history"]
    assert protocol["checkpoint_updates"] == list(POINTS)
    assert protocol["probe"]["source_updates"] == [0, 128]
    assert protocol["probe"]["updates"] == 16 and protocol["probe"]["checkpoint_updates"] == [
        0,
        4,
        8,
        16,
    ]
    assert protocol["seed"] == 17 and protocol["hidden_dim"] == 8 and protocol["num_classes"] == 7
    assert protocol["learning_rate"] == 0.03 and protocol["momentum"] == 0.9
    assert protocol["inference_steps"] == 2 and protocol["inference_learning_rate"] == 0.04
    work = protocol["planned_work"]
    for key, expected in dict(
        benchmark_native_train_calls=128,
        benchmark_training_rows=1152,
        probe_native_train_calls=32,
        probe_training_rows=288,
        native_train_calls=160,
        training_rows=1440,
        known_raw_score_forwards=30,
        known_raw_score_rows=270,
        probe_score_forwards=16,
        probe_score_rows=144,
        score_forwards=46,
        score_rows=414,
        teacher_episodes=10,
        teacher_decisions=90,
        maximum_source_environment_steps=160,
        learner_live_episodes=0,
        final_episodes=0,
        replay_rows=0,
        recurrent_unroll_steps=0,
    ).items():
        assert work[key] == expected
    if health:
        for key, expected in dict(
            activation_calls=30,
            activation_rows=270,
            activation_array_bytes=17280,
            activation_svd_calls=30,
            activation_feedforward_hidden_affines=30,
            activation_recurrent_hidden_steps=0,
            activation_recurrent_output_projections=0,
            activation_projection_rows=270,
            activation_projection_array_bytes=270 * 2368,
        ).items():
            assert work[key] == expected
    else:
        assert "activation_health" not in protocol
        assert not any(key.startswith("activation_") for key in work)
    json.dumps(protocol, allow_nan=False)


class StringSubclass(str):
    pass


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        0,
        1,
        0.5,
        [],
        {},
        "",
        "BP",
        "observation_memory",
        "public_invariant_bp",
        "recurrent_bc",
        StringSubclass("bp"),
    ],
)
def test_should_refuse_invalid_selector_types_or_unfrozen_methods_before_resources(
    monkeypatch, value
):
    monkeypatch.setattr(pilot, "require_torch", forbidden)
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    with pytest.raises(ValueError):
        pilot.describe_protocol(**HISTORY, qualification_method=value)
    with pytest.raises(ValueError):
        pilot.run_pilot(**HISTORY, qualification_method=value)


@pytest.mark.parametrize(
    "options",
    [
        dict(qualification_method="bp"),
        dict(public_history=True, qualification_method="bp"),
        dict(invariant_history=True, qualification_method="bp"),
        dict(public_history=1, invariant_history=True, qualification_method="bp"),
        dict(public_history=True, invariant_history=1, qualification_method="bp"),
        dict(**HISTORY, pc_qualification=True, qualification_method="pc"),
        dict(**HISTORY, pc_qualification=True, qualification_method="bp"),
        dict(**HISTORY, gradient_geometry=True, qualification_method="bp"),
    ],
)
def test_should_refuse_unsupported_or_conflicting_routes_before_resources(monkeypatch, options):
    monkeypatch.setattr(pilot, "require_torch", forbidden)
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    with pytest.raises(ValueError):
        pilot.describe_protocol(**options)
    with pytest.raises(ValueError):
        pilot.run_pilot(**options)


@pytest.mark.parametrize("health", [False, True])
def test_should_keep_selector_pc_identical_to_the_legacy_pc_descriptor(health):
    assert pilot.describe_protocol(
        **HISTORY, qualification_method="pc", activation_health=health
    ) == pilot.describe_protocol(**HISTORY, pc_qualification=True, activation_health=health)


@pytest.mark.parametrize(
    "options",
    [
        {},
        dict(public_history=True),
        HISTORY,
        dict(**HISTORY, pc_qualification=True),
        dict(**HISTORY, pc_qualification=True, activation_health=True),
    ],
)
def test_should_keep_none_selector_identical_to_every_existing_descriptor(options):
    assert pilot.describe_protocol(**options) == pilot.describe_protocol(
        **options, qualification_method=None
    )


@pytest.mark.parametrize("method", METHODS)
def test_should_construct_the_same_selected_native_family_at_zero_updates_without_learning(
    monkeypatch, method
):
    torch = require_torch()
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    monkeypatch.setattr(TorchFeatureLearner, "train_batch", forbidden)
    monkeypatch.setattr(InvariantTorchFeatureLearner, "train_batch", forbidden)
    torch_rng, numpy_rng = torch.get_rng_state().clone(), pickle.dumps(np.random.get_state())
    selected = f"public_invariant_{method}"
    source = pilot._history_learner(selected)
    assert type(source) is InvariantTorchFeatureLearner
    state = source.snapshot_state()
    assert (
        state.policy.method == selected
        and state.policy.encoding_id == INVARIANT_HISTORY_ENCODING_ID
    )
    expected = TorchFeaturePolicy(method, 296, 8, 17, num_classes=7)
    assert state.policy.native_policy == state.native_state.policy == expected
    assert state.completed_updates == state.training_examples == 0
    pilot.shared._invariant_feature_counter(state, 0, selected)
    native = state.native_state
    if method == "bp":
        assert native.optimizer_state is not None and native.gradients is not None
        assert native.optimizer_state["param_groups"][0]["lr"] == 0.03
        assert native.optimizer_state["param_groups"][0]["momentum"] == 0.9
    else:
        assert native.optimizer_state is native.gradients is None
    if method in ("neutral_cpc", "gated_cpc"):
        assert native.head_state["config"] == asdict(_config(expected))
    assert torch.equal(torch.get_rng_state(), torch_rng)
    assert pickle.dumps(np.random.get_state()) == numpy_rng


@pytest.mark.parametrize("method", METHODS)
def test_should_route_only_the_selected_method_through_actual_fit_and_both_paid_probe_helpers(
    monkeypatch, evaluator, method
):
    selected = f"public_invariant_{method}"
    case = environment(monkeypatch, evaluator, method=selected)
    report = pilot.run_pilot(**HISTORY, qualification_method=method)
    assert report["status"] == "complete", report["reason"]
    assert report["comparisons"] is None and len(report["methods"]) == 1
    row = report["methods"][0]
    assert row["method"] == selected and row["status"] == "complete"
    assert case.work["factory_calls"] == [selected]
    arm = case.work["arms"][0]
    assert arm["source"].policy.method == selected
    assert arm["source"].policy.native_policy.method == method
    assert arm["source"].state.completed_updates == arm["source"].state.training_examples == 0
    assert [point for point, _, _ in case.work["fit"]] == list(POINTS)
    assert [call["state"].completed_updates for call in case.work["probes"]] == [0, 128]
    assert row["charged_updates"] == arm["budget"].updates_completed == 128
    assert row["native_training_examples"] == 1152
    assert row["source_unchanged"] is row["actor_unchanged"] is True
    probes = case.work["probes"]
    assert probes[0]["budget"] is not probes[1]["budget"]
    for call in probes:
        assert call["budget"] is not arm["budget"] and call["budget"].updates_completed == 16
        assert call["source_before"] == call["source_after"]
        assert call["state_before"] == call["state_after"]
        assert call["inputs_before"] == call["inputs_after"]
        assert call["report"]["completed_training_examples"] == 144
        assert call["report"]["score_forward_calls_completed"] == 8
    assert len(case.native["trains"]) == 32 and len(case.native["predicts"]) == 46
    assert report["work"]["native_train_calls"] == 160 and report["work"]["training_rows"] == 1440
    assert report["work"]["teacher_episodes_observed"] == 10
    assert report["work"]["source_environment_steps_returned"] == 90
    assert row["initial_A_acquired"] is False
    assert row["interpretation"] == "inconclusive_initial_A_fit_incomplete"
    assert report["scientific_advantage_verified"] is False


def test_should_keep_a_failed_bp_probe_prefix_paid_and_stop_after_prior_evidence_corruption(
    monkeypatch, evaluator
):
    case = environment(monkeypatch, evaluator, method="public_invariant_bp")

    def corrupt(owner):
        if len(case.native.get("trains", [])) == 17:
            case.work["slots"][0]["report"]["completed_updates"] = 0

    case.native["on_train"] = corrupt
    report = pilot.run_pilot(**HISTORY, qualification_method="bp")
    assert report["status"] == "incomplete" and report["reason"]
    assert report["comparisons"] is None
    row = report["methods"][0]
    assert row["method"] == "public_invariant_bp"
    first, second = row["probes"]
    assert first["status"] == first["report"]["status"] == "complete"
    assert first["charged_updates"] == first["report"]["completed_updates"] == 16
    assert second["status"] == "failed"
    assert second["charged_updates"] == second["report"]["completed_updates"] == 1
    assert second["report"]["completed_training_examples"] == 9
    assert len(case.native["trains"]) == 17
    assert row["charged_updates"] == 128 and row["native_training_examples"] == 1152
    assert row["source_unchanged"] is row["actor_unchanged"] is True
    assert report["work"]["native_train_calls"] == 145 and report["work"]["training_rows"] == 1305
