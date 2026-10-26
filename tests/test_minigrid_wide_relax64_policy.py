"""Exact H32 fixed64 policy/clock gates, plus one separately budgeted native check.

Scripted checks construct no learner. The function with ``native`` in its name
performs exactly two TRAIN updates and two full-catalogue predictions; root must
select it separately under the prospective native allowance.
"""

from copy import deepcopy
from dataclasses import asdict, replace
import pickle
import random
from types import SimpleNamespace

import numpy as np
import pytest

from src.adapters import minigrid_learning_pilot as shared
from src.adapters import minigrid_relaxation_control as control
from src.adapters.invariant_torch_feature_learner import InvariantTorchFeaturePolicy
from src.adapters.torch_feature_learner import TorchFeaturePolicy, _config


@pytest.mark.parametrize("method", control.BOUND_METHODS)
def test_should_bind_explicit_h32_control_without_changing_original_h8_policy(method):
    base = method.removeprefix("public_invariant_").removesuffix("_relax64")
    original = control.policy(method)
    assert original == control.policy(method, wide_neural=False)
    assert original == InvariantTorchFeaturePolicy(
        TorchFeaturePolicy(base, 296, 8, 17, inference_steps=64, num_classes=7)
    )
    wide = control.policy(method, wide_neural=True)
    assert wide == replace(original, native_policy=replace(original.native_policy, hidden_dim=32))
    assert wide.method == f"public_invariant_{base}"
    assert wide.method != method


@pytest.mark.parametrize("flag", [None, 0, 1, "true", [], np.bool_(True)])
def test_should_refuse_nonprimitive_width_flags_before_policy_or_descriptor(flag):
    with pytest.raises(ValueError):
        control.policy(control.BOUND_METHODS[0], wide_neural=flag)
    with pytest.raises(ValueError):
        control.describe({}, control.METHODS[0], wide_neural=flag)


@pytest.mark.parametrize("method", ["pc", "neutral_cpc", "public_invariant_bp_relax64", None])
def test_should_refuse_unregistered_or_original_names_at_control_policy_boundary(method):
    with pytest.raises(ValueError):
        control.policy(method, wide_neural=True)


@pytest.mark.parametrize("method", control.METHODS)
def test_should_describe_actual_width_and_keep_nominal_work_separate(method):
    def protocol():
        return {
            "planned_work": {"native_train_calls": 192, "training_rows": 7488},
            "limitations": "original limitations",
        }

    original, explicit, wide = protocol(), protocol(), protocol()
    control.describe(original, method)
    control.describe(explicit, method, wide_neural=False)
    assert original == explicit
    control.describe(wide, method, wide_neural=True)
    recipe = wide["relaxation_control"]
    assert recipe["method"] == method
    assert recipe["native_policy"] == asdict(
        control.policy(f"public_invariant_{method}", wide_neural=True).native_policy
    )
    assert recipe["native_policy"]["hidden_dim"] == 32
    assert recipe["native_policy"]["inference_steps"] == 64
    assert recipe["original_inference_steps"] == 2
    assert wide["planned_work"]["relaxation_control"] == {
        "nominal_latent_iterations": 12288,
        "nominal_latent_row_iterations": 479232,
        "inference_steps": 64,
    }
    assert wide["planned_work"] == original["planned_work"]
    if method == "neutral_cpc_relax64":
        assert wide["cpc_configs"] == {
            method: asdict(
                _config(
                    control.policy(f"public_invariant_{method}", wide_neural=True).native_policy
                )
            )
        }
        assert "cpc_configs" not in original


def copied_state(monkeypatch, method):
    state = SimpleNamespace(
        policy=control.policy(method, wide_neural=True),
        completed_updates=2,
        training_examples=108,
    )
    monkeypatch.setattr(shared, "copy_invariant_torch_feature_snapshot", deepcopy)
    return state


@pytest.mark.parametrize("method", control.BOUND_METHODS)
def test_should_own_validated_wide_control_and_apply_original_54_times_point_law(
    monkeypatch, method
):
    state = copied_state(monkeypatch, method)
    owned = shared._invariant_feature_snapshot(
        state, 2, method, catalogue_rows=54, wide_neural=True
    )
    assert owned == state and owned is not state
    shared._invariant_feature_counter(state, 2, method, catalogue_rows=54, wide_neural=True)
    assert state.training_examples == 108
    for rows in (None, 9, 54.0, True):
        with pytest.raises(ValueError):
            shared._invariant_feature_snapshot(
                state, 2, method, catalogue_rows=rows, wide_neural=True
            )
    for point in (True, 2.0, -1, 129):
        with pytest.raises(ValueError):
            shared._invariant_feature_snapshot(
                state, point, method, catalogue_rows=54, wide_neural=True
            )


@pytest.mark.parametrize(
    "changes",
    [
        {"hidden_dim": 8},
        {"method": "neutral_cpc"},
        {"inference_steps": 2},
        {"seed": 18},
        {"learning_rate": 0.031},
        {"momentum": 0.8},
        {"inference_learning_rate": 0.041},
    ],
)
def test_should_refuse_entire_changed_native_recipe_under_wide_control_name(monkeypatch, changes):
    method = "public_invariant_pc_relax64"
    state = copied_state(monkeypatch, method)
    state.policy = replace(
        state.policy, native_policy=replace(state.policy.native_policy, **changes)
    )
    with pytest.raises(ValueError):
        shared._invariant_feature_snapshot(state, 2, method, catalogue_rows=54, wide_neural=True)


@pytest.mark.parametrize("method", control.BOUND_METHODS)
@pytest.mark.parametrize(
    "changes",
    [{"completed_updates": True}, {"completed_updates": 1}, {"training_examples": 18}],
)
def test_should_refuse_inexact_or_mismatched_control_counters(monkeypatch, method, changes):
    state = copied_state(monkeypatch, method)
    for name, value in changes.items():
        setattr(state, name, value)
    with pytest.raises(ValueError):
        shared._invariant_feature_snapshot(state, 2, method, catalogue_rows=54, wide_neural=True)


@pytest.mark.parametrize("method", control.BOUND_METHODS)
def test_should_refuse_h32_s64_relabeling_as_original_s2_or_h8_control(monkeypatch, method):
    state = copied_state(monkeypatch, method)
    with pytest.raises(ValueError):
        shared._invariant_feature_snapshot(state, 2, method, catalogue_rows=54)
    with pytest.raises(ValueError):
        shared._invariant_feature_snapshot(
            state, 2, method.removesuffix("_relax64"), catalogue_rows=54, wide_neural=True
        )


@pytest.mark.parametrize("method", control.BOUND_METHODS)
def test_should_admit_wide_control_only_to_original_joint_training_preflight(monkeypatch, method):
    class ReachedPacketValidation(Exception):
        pass

    def validate(*args):
        raise ReachedPacketValidation

    monkeypatch.setattr(shared, "_validate_batch", validate)
    options = {
        "learner_factory": lambda name: None,
        "catalogue_rows": 54,
        "wide_neural": True,
    }
    packets = (object(),) * 128
    with pytest.raises(ReachedPacketValidation):
        shared._train_arm({"method": method}, packets, None, 0.0, lambda: None, **options)
    for change in (
        {"learner_factory": None},
        {"catalogue_rows": None},
        {"catalogue_rows": 54.0},
        {"checkpoint_updates": (0, 32)},
        {"public_history": True},
        {"invariant_fifo36": True},
        {"wide_neural": 1},
    ):
        with pytest.raises(ValueError):
            shared._train_arm(
                {"method": method}, packets, None, 0.0, lambda: None, **{**options, **change}
            )
    with pytest.raises(ValueError):
        shared._train_arm({"method": method}, packets[:32], None, 0.0, lambda: None, **options)


def test_native_should_preserve_h32_fixed64_pc_neutral_parity_in_two_train_updates():
    """Exactly108 TRAIN rows and108 prediction rows, no physics or final data."""
    from src.adapters import minigrid_learnability as pilot
    from src.adapters.torch_feature_rehearsal_pilot import _same_state
    from src.shared.torch_runtime import require_torch
    from test_minigrid_relax64_native import training_sources

    torch = require_torch()
    global_before = (
        random.getstate(),
        pickle.dumps(np.random.get_state()),
        torch.get_rng_state().clone(),
    )
    sources = training_sources()
    plan = pilot.joint.compose(pilot._history_packets(sources))
    batch = plan.training[0]
    inputs = pickle.dumps((sources, batch.features, batch.targets))
    owners = [pilot._history_learner(method, wide_neural=True) for method in control.BOUND_METHODS]
    originals = [owner.snapshot_state() for owner in owners]
    candidates = [owner.fork() for owner in owners]
    names = ("weight_feature_hidden", "bias_hidden", "weight_hidden_output", "bias_output")
    for name in names:
        assert torch.equal(
            originals[0].native_state.head_state[name], originals[1].native_state.head_state[name]
        )
    finals, predictions = [], []
    for method, candidate in zip(control.BOUND_METHODS, candidates, strict=True):
        candidate.train_batch(batch.features, batch.targets)
        state = candidate.snapshot_state()
        assert state.policy == control.policy(method, wide_neural=True)
        assert state.completed_updates == 1 and state.training_examples == 54
        owned = shared._invariant_feature_snapshot(
            state, 1, method, catalogue_rows=54, wide_neural=True
        )
        assert _same_state(owned, state)
        for name in names:
            assert (
                owned.native_state.head_state[name].data_ptr()
                != state.native_state.head_state[name].data_ptr()
            )
        with pytest.raises(ValueError):
            shared._invariant_feature_counter(
                state, 1, method.removesuffix("_relax64"), catalogue_rows=54, wide_neural=True
            )
        with pytest.raises(ValueError):
            shared._invariant_feature_counter(state, 1, method, catalogue_rows=54)
        predictions.append(candidate.predict(batch.features))
        assert _same_state(state, candidate.snapshot_state())
        finals.append(state.native_state.head_state)
    assert any(
        not torch.equal(originals[0].native_state.head_state[name], finals[0][name])
        for name in names
    )
    for name in names:
        assert torch.equal(finals[0][name], finals[1][name])
    np.testing.assert_array_equal(predictions[0], predictions[1])
    for owner, original in zip(owners, originals, strict=True):
        assert _same_state(owner.snapshot_state(), original)
    assert pickle.dumps((sources, batch.features, batch.targets)) == inputs
    assert random.getstate() == global_before[0]
    assert pickle.dumps(np.random.get_state()) == global_before[1]
    assert torch.equal(torch.get_rng_state(), global_before[2])
