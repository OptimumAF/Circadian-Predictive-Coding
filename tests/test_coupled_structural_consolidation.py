"""Tagged split/merge/prune consolidation with private inputs and spent attempts."""

from dataclasses import asdict, replace
from typing import Any

import numpy as np
import pytest

import src.app.coupled_structural_consolidation as consolidation
import src.app.coupled_scaling_prune_consolidation as old_prune
from src.adapters.structural_coupled_scaling_learner import StructuralCoupledScalingLearner
from src.app.actor_shadow import ActorShadowRuntime
from src.app.coupled_pruning_gate import CoupledPruneLimits, CoupledPruneDiagnostic
from src.app.coupled_scaling_pruning_gate import CoupledScalingPruneDecision
from src.app.coupled_structural_consolidation import (
    CoupledScalingSplitConsolidator,
    CoupledScalingMergeConsolidator,
    CoupledScalingStructuralPruneConsolidator,
)
from src.app.linear_scaling_measurement import ScalingBatch
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.actor_ports import ConsolidatedState
from src.core.coupled_circadian_predictive_coding import CoupledCircadianConfig
from src.core.coupled_inference import CoupledInferenceDiagnostics
from src.core.coupled_pc_scaling import (
    CoupledScalingConfig,
    copy_coupled_scaling_state,
    effective_coupled_parameters,
    initialize_coupled_scaling,
    step_coupled_scaling,
)
from src.core.coupled_scaling_pruning import prune_coupled_scaling_width
from src.core.coupled_scaling_structure import (
    split_coupled_scaling_width,
    merge_coupled_scaling_width,
)
from src.core.coupled_scaling_topology import CoupledScalingTopologyState
from src.core.coupled_structural_state import CoupledStructuralState, CoupledStructuralTransition
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.learner_ports import TrainingDiagnostic


def raw(family="pc"):
    policy = (
        None
        if family == "pc"
        else (
            CoupledCircadianConfig.matched_pc_control()
            if family == "neutral"
            else CoupledCircadianConfig(0.9, 0.2, 0.7, 0.2)
        )
    )
    initial = initialize_coupled_scaling(
        CoupledScalingConfig("mean_field", 2, 3, 2, 2.0, precisions=(0.7, 0.7), circadian=policy),
        41,
    )
    if initial.wake_state is not None:
        chemical = np.array([0.3, 0.7, 1.1])
        assert policy is not None
        initial = replace(
            initial,
            wake_state=replace(
                initial.wake_state,
                chemical_state=chemical,
                plasticity_state=np.clip(
                    np.exp(-policy.plasticity_sensitivity * chemical), policy.min_plasticity, 1
                ),
                wake_updates=4,
            ),
        )
    return initial


def envelope(family="pc"):
    source = raw(family)
    return CoupledStructuralState(1, source.config, source)


def probes():
    return (
        ScalingBatch("train", ("local-train",), np.array([[0.2, -0.3]]), np.array([[0.2]])),
        ScalingBatch("validation", ("local-guard",), np.array([[0.7, 0.4]]), np.array([[0.8]])),
    )


def budget(updates=0):
    return ToyBudgetSession(ToyExecutionBudget(max_training_updates=updates), lambda: 0.0)


def prune(data=None, **changes):
    settings: dict[str, Any] = dict(
        limits=CoupledPruneLimits(1, 1, 1),
        budget=budget(),
        inference_steps=2,
        inference_learning_rate=0.01,
    )
    settings.update(changes)
    return CoupledScalingStructuralPruneConsolidator(
        (1, 1), probes() if data is None else data, **settings
    )


def evidence(data):
    source = CoupledInferenceDiagnostics("max_steps", 2, 2, 3, 3, 2, 0.6, 0.5, 0.1, False)
    candidate = replace(source, initial_energy=0.7, final_energy=0.6)
    return tuple(
        CoupledPruneDiagnostic(
            p.role,
            p.row_ids,
            0.5,
            0.6,
            0.5,
            0.6,
            0.1 * (i + 1),
            0.6 - 0.5,
            0.6 - 0.5,
            source,
            candidate,
        )
        for i, p in enumerate(data)
    )


def fake_gate(state, removed, data, **kwargs):
    changed = prune_coupled_scaling_width(state, removed)
    return CoupledScalingPruneDecision(
        "accepted",
        None,
        changed.state,
        changed.lineage,
        changed.max_effective_relative_error,
        evidence(data),
        len(data),
    )


def runtime(initial=None):
    source = StructuralCoupledScalingLearner(
        raw() if initial is None else initial,
        learning_rate=0.03,
        inference_steps=2,
        inference_learning_rate=0.05,
    )
    clock = LogicalClock()
    session = budget(1)
    rt = ActorShadowRuntime(
        source,
        actor_version="a0",
        candidate_version="c0",
        clock=clock,
        budget=session,
        max_experiences=1,
        max_consolidations=8,
    )
    return rt, clock, session, source


@pytest.mark.parametrize("kind", ("split", "merge"))
def test_should_copy_inputs_kernel_results_and_constructor_lineage_before_appending_owned_tags(
    monkeypatch, kind
):
    origin = envelope("active")
    split = split_coupled_scaling_width(origin.raw_state)
    incoming = (
        origin
        if kind == "split"
        else CoupledStructuralState(
            1,
            origin.origin_config,
            split.state,
            (CoupledStructuralTransition("split", split.lineage),),
        )
    )
    before = asdict(incoming)
    calls = []

    def kernel(state, *args):
        result = (
            split_coupled_scaling_width(state)
            if kind == "split"
            else merge_coupled_scaling_width(state, args[0])
        )
        calls.append(result)
        state.raw_parameters.output_bias[:] = 999
        if args:
            object.__setattr__(args[0].source_config, "gamma0", 3.0)
        return result

    monkeypatch.setattr(consolidation, kind + "_coupled_scaling_width", kernel)
    operation = (
        CoupledScalingSplitConsolidator()
        if kind == "split"
        else CoupledScalingMergeConsolidator(split.lineage)
    )
    if kind == "merge":
        object.__setattr__(split.lineage.source_config, "gamma0", 4.0)
    a = operation(incoming)
    b = operation(incoming)
    np.testing.assert_equal(asdict(a), asdict(b))
    assert a.state.transitions[-1].kind == kind
    assert a.diagnostic.definition == f"coupled_scaling_{kind}_effective_encoding_relative_error_v1"
    assert 0 <= a.diagnostic.value <= 8 * np.finfo(float).eps
    calls[0].state.raw_parameters.output_bias[:] = 998
    assert a.state.raw_state.raw_parameters.output_bias[0, 0] != 998
    np.testing.assert_equal(asdict(incoming), before)


@pytest.mark.parametrize("kind", ("split", "merge"))
@pytest.mark.parametrize(
    "bad",
    (
        "family",
        "config",
        "lineage",
        "nanerror",
        "largeerror",
        "negativeerror",
        "dtype",
        "count",
        "chemistry",
    ),
)
def test_should_refuse_malformed_trusted_width_change_before_structural_publication(
    kind, bad, monkeypatch
):
    original = envelope("active")
    split = split_coupled_scaling_width(original.raw_state)
    state = (
        original
        if kind == "split"
        else CoupledStructuralState(
            1,
            original.origin_config,
            split.state,
            (CoupledStructuralTransition("split", split.lineage),),
        )
    )
    before = asdict(state)

    def malformed(raw, *args):
        result = (
            split_coupled_scaling_width(raw)
            if kind == "split"
            else merge_coupled_scaling_width(raw, args[0])
        )
        if bad == "family":
            return asdict(result)
        if bad == "config":
            object.__setattr__(result.state.config, "gamma0", 3.0)
        elif bad == "lineage":
            object.__setattr__(result.lineage.source_config, "gamma0", 3.0)
        elif bad == "nanerror":
            return replace(result, max_effective_relative_error=float("nan"))
        elif bad == "largeerror":
            return replace(result, max_effective_relative_error=1.0)
        elif bad == "negativeerror":
            return replace(result, max_effective_relative_error=-1.0)
        elif bad == "dtype":
            object.__setattr__(
                result.state.raw_parameters, "output_bias", np.zeros((1, 1), dtype=np.float32)
            )
        elif bad == "count":
            assert result.state.wake_state is not None
            object.__setattr__(result.state.wake_state, "wake_updates", 5)
        else:
            wake = result.state.wake_state
            assert wake is not None and result.state.config.circadian is not None
            wake.chemical_state[0] += 0.1
            wake.plasticity_state[0] = np.clip(
                np.exp(
                    -result.state.config.circadian.plasticity_sensitivity * wake.chemical_state[0]
                ),
                result.state.config.circadian.min_plasticity,
                1,
            )
        return result

    monkeypatch.setattr(consolidation, kind + "_coupled_scaling_width", malformed)
    operation = (
        CoupledScalingSplitConsolidator()
        if kind == "split"
        else CoupledScalingMergeConsolidator(split.lineage)
    )
    with pytest.raises(ValueError):
        operation(state)
    np.testing.assert_equal(asdict(state), before)


def test_should_allow_declared_non_lifo_compatible_merge_without_recovering_historical_parameters():
    original = raw()
    split = split_coupled_scaling_width(original)
    split.state.raw_parameters.output_bias[:] = 0.25
    current_bias = effective_coupled_parameters(split.state).output_bias.copy()
    rooted_here = CoupledStructuralState(1, split.state.config, split.state)
    result = CoupledScalingMergeConsolidator(split.lineage)(rooted_here)
    assert [t.kind for t in result.state.transitions] == ["merge"]
    assert (
        result.state.origin_config == split.state.config
        and result.state.raw_state.config == original.config
    )
    np.testing.assert_allclose(
        effective_coupled_parameters(result.state.raw_state).output_bias,
        current_bias,
        rtol=8 * np.finfo(float).eps,
        atol=0,
    )
    assert original.raw_parameters.output_bias[0, 0] == 0


def test_should_wrap_prune_with_captured_full_mixed_history_and_temporary_empty_prune_metadata(
    monkeypatch,
):
    source = envelope("active")
    split = CoupledScalingSplitConsolidator()(source).state
    before = asdict(split)
    captured = []

    def accepted_inner(self, temp):
        assert type(temp) is CoupledScalingTopologyState and temp.prune_lineage == ()
        changed = prune_coupled_scaling_width(temp.raw_state, (1, 1))
        result = ConsolidatedState(
            CoupledScalingTopologyState(1, changed.state, (changed.lineage,)),
            TrainingDiagnostic("coupled_scaling_prune_max_probability_change_v1", 0.2),
        )
        captured.append(result)
        temp.raw_state.raw_parameters.output_bias[:] = 999
        return result

    monkeypatch.setattr(old_prune.CoupledScalingPruneConsolidator, "__call__", accepted_inner)
    result = prune()(split)
    assert [t.kind for t in result.state.transitions] == ["split", "prune"]
    assert result.state.origin_config == source.origin_config
    assert result.diagnostic.definition == "coupled_scaling_prune_max_probability_change_v1"
    captured[0].state.raw_state.raw_parameters.output_bias[:] = 998
    assert result.state.raw_state.raw_parameters.output_bias[0, 0] != 998
    np.testing.assert_equal(asdict(split), before)


@pytest.mark.parametrize(
    "bad", ("family", "history_empty", "history_extra", "lineage", "definition", "value", "count")
)
def test_should_refuse_malformed_existing_prune_transform_result_without_dropping_mixed_history(
    bad, monkeypatch
):
    source = CoupledScalingSplitConsolidator()(envelope("active")).state
    before = asdict(source)

    def malformed(self, temp):
        change = prune_coupled_scaling_width(temp.raw_state, (1, 1))
        state = CoupledScalingTopologyState(1, change.state, (change.lineage,))
        diagnostic = TrainingDiagnostic("coupled_scaling_prune_max_probability_change_v1", 0.2)
        if bad == "family":
            return asdict(state)
        if bad == "history_empty":
            object.__setattr__(state, "prune_lineage", ())
        elif bad == "history_extra":
            object.__setattr__(state, "prune_lineage", state.prune_lineage * 2)
        elif bad == "lineage":
            object.__setattr__(state.prune_lineage[0], "removed_units", (0, 0))
        elif bad == "definition":
            diagnostic = TrainingDiagnostic("other_v1", 0.2)
        elif bad == "value":
            diagnostic = TrainingDiagnostic("coupled_scaling_prune_max_probability_change_v1", 1.1)
        else:
            assert state.raw_state.wake_state is not None
            object.__setattr__(state.raw_state.wake_state, "wake_updates", 5)
        return ConsolidatedState(state, diagnostic)

    monkeypatch.setattr(old_prune.CoupledScalingPruneConsolidator, "__call__", malformed)
    with pytest.raises(ValueError):
        prune()(source)
    np.testing.assert_equal(asdict(source), before)


def test_should_retain_split_prune_tags_and_spend_incompatible_inverse_attempt_before_merge_kernel(
    monkeypatch,
):
    monkeypatch.setattr(old_prune, "gate_coupled_scaling_prune", fake_gate)
    rt, _, _, source = runtime(raw("active"))
    actor = rt.actor.snapshot_state()
    rt.consolidate("split", CoupledScalingSplitConsolidator())
    lineage = rt.candidate_snapshot().state.transitions[-1].lineage
    rt.consolidate("prune", prune())
    before = rt.candidate_snapshot()

    def forbidden(*args, **kwargs):
        pytest.fail("incompatible inverse called merge kernel")

    monkeypatch.setattr(consolidation, "merge_coupled_scaling_width", forbidden)
    with pytest.raises(ValueError):
        rt.consolidate("inverse", CoupledScalingMergeConsolidator(lineage))
    after = rt.candidate_snapshot()
    assert after.consolidation_attempts == 3 and after.consolidations_completed == 2
    assert [t.kind for t in after.state.transitions] == ["split", "prune"]
    np.testing.assert_equal(asdict(after.state), asdict(before.state))
    np.testing.assert_equal(asdict(rt.actor.snapshot_state()), asdict(actor))
    recovered = rt.consolidate("recover", CoupledScalingSplitConsolidator())
    assert recovered.attempt_number == 4
    assert rt.candidate_snapshot().consolidations_completed == 3
    assert [t.kind for t in rt.candidate_snapshot().state.transitions] == [
        "split",
        "prune",
        "split",
    ]
    assert source.snapshot_state().transitions == ()


@pytest.mark.parametrize("status", ("rejected", "failed"))
def test_should_propagate_owned_prune_refusal_and_spend_attempt_with_complete_mixed_state_intact(
    status, monkeypatch
):
    def refuse(state, removed, data, **kwargs):
        return CoupledScalingPruneDecision(
            status,
            "injected",
            copy_coupled_scaling_state(state),
            None,
            None,
            evidence(data) if status == "rejected" else evidence(data)[:1],
            len(data),
        )

    monkeypatch.setattr(old_prune, "gate_coupled_scaling_prune", refuse)
    rt, _, _, _ = runtime(raw("active"))
    actor = rt.actor.snapshot_state()
    rt.consolidate("split", CoupledScalingSplitConsolidator())
    before = rt.candidate_snapshot()
    with pytest.raises(old_prune.CoupledScalingPruneRefused) as caught:
        rt.consolidate("refuse", prune())
    after = rt.candidate_snapshot()
    assert after.consolidation_attempts == 2 and after.consolidations_completed == 1
    np.testing.assert_equal(asdict(after.state), asdict(before.state))
    detached = caught.value.decision
    detached.state.raw_parameters.output_bias[:] = 999
    np.testing.assert_equal(asdict(after.state), asdict(rt.candidate_snapshot().state))
    np.testing.assert_equal(asdict(rt.actor.snapshot_state()), asdict(actor))


@pytest.mark.parametrize("kind", ("split", "merge", "prune"))
def test_should_refuse_full_history_before_any_transform_or_gate_callback(kind, monkeypatch):
    source = raw()
    first = split_coupled_scaling_width(source)
    current = first.state
    second = split_coupled_scaling_width(current)
    pair = (
        CoupledStructuralTransition("split", second.lineage),
        CoupledStructuralTransition("merge", second.lineage),
    )
    full = CoupledStructuralState(1, current.config, current, pair * 32)

    def forbidden(*args, **kwargs):
        pytest.fail("exhausted history reached callback")

    monkeypatch.setattr(consolidation, "split_coupled_scaling_width", forbidden)
    monkeypatch.setattr(consolidation, "merge_coupled_scaling_width", forbidden)
    monkeypatch.setattr(old_prune, "gate_coupled_scaling_prune", forbidden)
    operation = (
        CoupledScalingSplitConsolidator()
        if kind == "split"
        else (CoupledScalingMergeConsolidator(first.lineage) if kind == "merge" else prune())
    )
    with pytest.raises(ValueError):
        operation(full)


def test_should_preserve_existing_prune_constructor_final_role_refusal_before_payload_copy():
    class Sentinel:
        def __deepcopy__(self, memo):
            raise AssertionError("final data copied")

    a, b = probes()
    with pytest.raises(ValueError):
        prune((replace(a, role="final_test", features=Sentinel(), targets=Sentinel()), b))


def test_should_run_real_three_family_split_merge_prune_then_delayed_candidate_wake_with_raw_reference():
    results = {}
    for family in ("pc", "neutral", "active"):
        source_raw = raw(family)
        for layer, (w, b) in enumerate(
            zip(
                source_raw.raw_parameters.hidden_weights,
                source_raw.raw_parameters.hidden_biases,
                strict=True,
            )
        ):
            w[:, 1] = 0
            b[:, 1] = 0
            if layer:
                w[1, :] = 0
        source_raw.raw_parameters.output_weight[1, :] = 0
        rt, clock, session, source = runtime(source_raw)
        original, actor = source.snapshot_state(), rt.actor.snapshot_state()
        rt.consolidate("split", CoupledScalingSplitConsolidator())
        lineage = rt.candidate_snapshot().state.transitions[-1].lineage
        rt.consolidate("merge", CoupledScalingMergeConsolidator(lineage))
        rt.consolidate("prune", prune(limits=CoupledPruneLimits(1e-12, 1e-12, 1e-12)))
        before = rt.candidate_snapshot().state
        assert [t.kind for t in before.transitions] == ["split", "merge", "prune"]
        assert before.raw_state.config.width == 2
        x, y = np.array([[0.3, -0.4], [0.7, 0.2]]), np.array([[0.0], [0.35]])
        prediction = rt.actor.predict(x).prediction
        rt.record_experience(
            Experience(
                "new-row", "episode", 0, "a0", x, "train", ExperiencePermissions(training=True)
            )
        )
        rt.record_label(LabelArrival("new-label", "new-row", "episode", 1, "a0", y))
        assert rt.train_ready() == ()
        clock.advance_to(1)
        applied = rt.train_ready(max_updates=1)
        reference = step_coupled_scaling(before.raw_state, x, y, 0.03, 2, 0.05)
        assert len(applied) == 1 and applied[0].diagnostic.value == reference.evaluation.energy
        candidate = rt.candidate_snapshot()
        np.testing.assert_equal(asdict(candidate.state.raw_state), asdict(reference.state))
        assert candidate.state.transitions == before.transitions
        assert candidate.state.origin_config == original.origin_config
        assert candidate.wake_updates == session.updates_completed == 1
        np.testing.assert_equal(asdict(rt.actor.snapshot_state()), asdict(actor))
        np.testing.assert_array_equal(rt.actor.predict(x).prediction, prediction)
        np.testing.assert_equal(asdict(source.snapshot_state()), asdict(original))
        results[family] = candidate.state.raw_state
    np.testing.assert_equal(
        asdict(results["pc"].raw_parameters), asdict(results["neutral"].raw_parameters)
    )
