"""Raw pruning through owned consolidation, without serving authority or final data."""

from copy import deepcopy
from dataclasses import asdict, replace
import pickle
from typing import Any

import numpy as np
import pytest

import src.app.coupled_scaling_prune_consolidation as consolidation
from src.adapters.topology_coupled_scaling_learner import TopologyCoupledScalingLearner
from src.app.actor_shadow import ActorShadowRuntime
from src.app.coupled_pruning_gate import CoupledPruneDiagnostic, CoupledPruneLimits
from src.app.coupled_scaling_prune_consolidation import (
    CoupledScalingPruneConsolidator,
    CoupledScalingPruneRefused,
)
from src.app.coupled_scaling_pruning_gate import CoupledScalingPruneDecision
from src.app.linear_scaling_measurement import ScalingBatch
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.actor_ports import ConsolidatedState
from src.core.coupled_circadian_predictive_coding import CoupledCircadianConfig
from src.core.coupled_inference import CoupledInferenceDiagnostics
from src.core.coupled_pc_scaling import (
    CoupledScalingConfig,
    copy_coupled_scaling_state,
    initialize_coupled_scaling,
    step_coupled_scaling,
)
from src.core.coupled_scaling_pruning import CoupledScalingPruneLineage, prune_coupled_scaling_width
from src.core.coupled_scaling_topology import CoupledScalingTopologyState
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.learner_ports import TrainingDiagnostic


def raw(family="pc", width=3, hidden=2):
    policy = (
        None
        if family == "pc"
        else (
            CoupledCircadianConfig.matched_pc_control()
            if family == "neutral"
            else CoupledCircadianConfig(0.9, 0.2, 0.7, 0.2)
        )
    )
    state = initialize_coupled_scaling(
        CoupledScalingConfig(
            "mean_field", 2, width, hidden, 2.0, precisions=(0.7,) * hidden, circadian=policy
        ),
        41,
    )
    if state.wake_state is not None:
        chemical = np.linspace(0.3, 1.1, width)
        assert policy is not None
        state = replace(
            state,
            wake_state=replace(
                state.wake_state,
                chemical_state=chemical,
                plasticity_state=np.clip(
                    np.exp(-policy.plasticity_sensitivity * chemical), policy.min_plasticity, 1
                ),
                wake_updates=4,
            ),
        )
    return state


def probes():
    return (
        ScalingBatch("train", ("local-train",), np.array([[0.2, -0.3]]), np.array([[0.2]])),
        ScalingBatch("validation", ("local-guard",), np.array([[0.7, 0.4]]), np.array([[0.8]])),
    )


def budget(updates=0):
    return ToyBudgetSession(ToyExecutionBudget(max_training_updates=updates), lambda: 0.0)


def operation(data=None, **changes):
    settings: dict[str, Any] = dict(
        limits=CoupledPruneLimits(1, 1, 1),
        budget=budget(),
        inference_steps=2,
        inference_learning_rate=0.01,
    )
    settings.update(changes)
    return CoupledScalingPruneConsolidator((1, 1), probes() if data is None else data, **settings)


def diagnostics(data):
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


def accepted(state, removed, data):
    change = prune_coupled_scaling_width(state, removed)
    return CoupledScalingPruneDecision(
        "accepted",
        None,
        change.state,
        change.lineage,
        change.max_effective_relative_error,
        diagnostics(data),
        len(data),
    )


def runtime(initial=None):
    source = TopologyCoupledScalingLearner(
        raw() if initial is None else initial,
        learning_rate=0.03,
        inference_steps=2,
        inference_learning_rate=0.05,
    )
    clock = LogicalClock()
    session = budget(1)
    return (
        ActorShadowRuntime(
            source,
            actor_version="a0",
            candidate_version="c0",
            clock=clock,
            budget=session,
            max_experiences=1,
            max_consolidations=3,
        ),
        clock,
        session,
        source,
    )


def test_should_append_owned_history_and_copy_gate_inputs_on_each_invocation(monkeypatch):
    first = prune_coupled_scaling_width(raw("active", width=4), (0, 2))
    wrapper = CoupledScalingTopologyState(1, first.state, (first.lineage,))
    before = asdict(wrapper)
    data = probes()
    limits = CoupledPruneLimits(1, 1, 1)
    callback = operation(data, limits=limits)
    for p in data:
        p.features[:] = 99
        p.targets[:] = 99
    object.__setattr__(limits, "max_prediction_change", 0.0)
    calls = []

    def fake_gate(state, removed, batches, *, limits, **kwargs):
        calls.append(1)
        assert limits.max_prediction_change == 1
        for actual, expected in zip(batches, probes(), strict=True):
            np.testing.assert_array_equal(actual.features, expected.features)
            np.testing.assert_array_equal(actual.targets, expected.targets)
        result = accepted(state, removed, batches)
        state.raw_parameters.output_bias[:] = 999
        for p in batches:
            p.features[:] = 999
            p.targets[:] = 999
        object.__setattr__(limits, "max_prediction_change", 0.0)
        return result

    monkeypatch.setattr(consolidation, "gate_coupled_scaling_prune", fake_gate)
    a, b = callback(wrapper), callback(wrapper)
    assert calls == [1, 1]
    np.testing.assert_equal(asdict(a), asdict(b))
    assert a.state.raw_state.config.width == 2 and len(a.state.prune_lineage) == 2
    assert a.state.prune_lineage[0] == first.lineage
    assert (
        a.diagnostic.definition == "coupled_scaling_prune_max_probability_change_v1"
        and a.diagnostic.value == 0.2
    )
    np.testing.assert_equal(asdict(wrapper), before)


@pytest.mark.parametrize("status", ("rejected", "failed"))
def test_should_return_fresh_owned_refusal_decisions_and_support_pickle_deepcopy(status):
    state = raw("active")
    evidence = diagnostics(probes()) if status == "rejected" else diagnostics(probes())[:1]
    decision = CoupledScalingPruneDecision(status, "injected", state, None, None, evidence, 2)
    refusal = CoupledScalingPruneRefused(decision)
    state.raw_parameters.output_bias[:] = 99
    assert refusal.decision.state.raw_parameters.output_bias[0, 0] == 0
    detached = refusal.decision
    detached.state.raw_parameters.output_bias[:] = 88
    assert detached.state.wake_state is not None
    detached.state.wake_state.chemical_state[:] = 88
    assert refusal.decision.state.raw_parameters.output_bias[0, 0] == 0
    for clone in (deepcopy(refusal), pickle.loads(pickle.dumps(refusal))):
        np.testing.assert_equal(asdict(clone.decision), asdict(refusal.decision))


@pytest.mark.parametrize("status", ("rejected", "failed"))
def test_should_charge_refused_runtime_attempt_without_candidate_actor_or_source_publication(
    monkeypatch, status
):
    def refuse(state, removed, data, **kwargs):
        return CoupledScalingPruneDecision(
            status,
            "injected refusal",
            copy_coupled_scaling_state(state),
            None,
            None,
            diagnostics(data) if status == "rejected" else diagnostics(data)[:1],
            len(data),
        )

    monkeypatch.setattr(consolidation, "gate_coupled_scaling_prune", refuse)
    rt, _, _, source = runtime(raw("active"))
    before, actor = rt.candidate_snapshot(), rt.actor.snapshot_state()
    with pytest.raises(CoupledScalingPruneRefused):
        rt.consolidate("refused", operation())
    after = rt.candidate_snapshot()
    np.testing.assert_equal(asdict(after.state), asdict(before.state))
    np.testing.assert_equal(asdict(rt.actor.snapshot_state()), asdict(actor))
    np.testing.assert_equal(asdict(source.snapshot_state()), asdict(before.state))
    assert after.consolidation_attempts == 1 and after.consolidations_completed == 0
    receipt = rt.consolidate(
        "next", lambda state: ConsolidatedState(state, TrainingDiagnostic("noop_v1", 0.0))
    )
    assert receipt.attempt_number == 2


def test_should_charge_gate_fault_and_leave_runtime_candidate_available(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("gate fault")

    monkeypatch.setattr(consolidation, "gate_coupled_scaling_prune", fail)
    rt, _, _, _ = runtime()
    before, actor = rt.candidate_snapshot(), rt.actor.snapshot_state()
    with pytest.raises(RuntimeError, match="gate fault"):
        rt.consolidate("fault", operation())
    after = rt.candidate_snapshot()
    assert after.consolidation_attempts == 1 and after.consolidations_completed == 0
    np.testing.assert_equal(asdict(after.state), asdict(before.state))
    np.testing.assert_equal(asdict(rt.actor.snapshot_state()), asdict(actor))


@pytest.mark.parametrize(
    "bad",
    (
        "family",
        "status",
        "lineage_none",
        "lineage_wrong",
        "config",
        "nanerror",
        "largeerror",
        "empty",
        "planned",
        "role",
        "row_ids",
        "wake",
        "early",
        "fallback",
        "energy_counter",
        "gradient_counter",
        "comparison_counter",
        "nonfinite",
        "energy_mismatch",
        "energy_increase",
        "inconsistent_bce",
        "inconsistent_energy",
        "probability",
        "inference_family",
    ),
)
def test_should_reject_malformed_accepted_gate_evidence_before_replacement(monkeypatch, bad):
    wrapper = CoupledScalingTopologyState(1, raw("active"))
    before = asdict(wrapper)

    def malformed(state, removed, data, **kwargs):
        result = accepted(state, removed, data)
        first = result.diagnostics[0]
        inference_changes: dict[str, dict[str, Any]] = {
            "early": {"reason": "gradient_tolerance", "attempted_steps": 1, "accepted_steps": 1},
            "fallback": {"used_fallback": True},
            "energy_counter": {"energy_evaluations": 2},
            "gradient_counter": {"gradient_norm_evaluations": 2},
            "comparison_counter": {"energy_comparisons": 1},
            "nonfinite": {"final_gradient_inf_norm": float("nan")},
            "energy_mismatch": {"final_energy": 0.4},
            "energy_increase": {"initial_energy": 0.4},
        }
        if bad in inference_changes:
            first = replace(
                first, source_inference=replace(first.source_inference, **inference_changes[bad])
            )
            return replace(result, diagnostics=(first, result.diagnostics[1]))
        if bad in ("inconsistent_bce", "inconsistent_energy", "probability", "inference_family"):
            changes: dict[str, dict[str, Any]] = {
                "inconsistent_bce": {"bce_increase": 0.1},
                "inconsistent_energy": {"inferred_energy_increase": 0.1},
                "probability": {"max_prediction_change": 1.1},
                "inference_family": {"source_inference": asdict(first.source_inference)},
            }
            return replace(
                result, diagnostics=(replace(first, **changes[bad]), result.diagnostics[1])
            )
        if bad == "family":
            return asdict(result)
        if bad == "status":
            return replace(result, status="best")
        if bad == "lineage_none":
            return replace(result, lineage=None)
        if bad == "lineage_wrong":
            return replace(result, lineage=replace(result.lineage, removed_units=(0, 0)))
        if bad == "config":
            object.__setattr__(result.state.config, "width", 3)
        elif bad == "nanerror":
            return replace(result, max_effective_relative_error=float("nan"))
        elif bad == "largeerror":
            return replace(result, max_effective_relative_error=1e-3)
        elif bad == "empty":
            return replace(result, diagnostics=())
        elif bad == "planned":
            return replace(result, planned_probe_count=1)
        elif bad == "role":
            return replace(
                result,
                diagnostics=(
                    replace(result.diagnostics[0], role="final_test"),
                    result.diagnostics[1],
                ),
            )
        elif bad == "row_ids":
            return replace(
                result,
                diagnostics=(
                    replace(result.diagnostics[0], row_ids=("wrong",)),
                    result.diagnostics[1],
                ),
            )
        elif bad == "wake":
            assert result.state.wake_state is not None
            object.__setattr__(result.state.wake_state, "wake_updates", 0)
        return result

    monkeypatch.setattr(consolidation, "gate_coupled_scaling_prune", malformed)
    with pytest.raises(ValueError):
        operation()(wrapper)
    np.testing.assert_equal(asdict(wrapper), before)


def test_should_refuse_history_exhaustion_before_gate_or_budget_callback(monkeypatch):
    state = raw()
    records = tuple(
        CoupledScalingPruneLineage(1, replace(state.config, width=67 - i), (0, 0))
        for i in range(64)
    )
    wrapper = CoupledScalingTopologyState(1, state, records)

    def forbidden(*args, **kwargs):
        pytest.fail("history overflow reached gate or budget")

    monkeypatch.setattr(consolidation, "gate_coupled_scaling_prune", forbidden)
    session = budget()
    monkeypatch.setattr(session, "before_final", forbidden)
    with pytest.raises(ValueError):
        operation(budget=session)(wrapper)


def invalid_data():
    a, b = probes()
    return [
        (),
        [a, b],
        (a, b) * 5,
        (a,),
        (replace(a, role="final_test"), b),
        (replace(a, row_ids=(" padded ",)), b),
        (replace(a, row_ids=("line\nbreak",)), b),
        (a, replace(b, row_ids=a.row_ids)),
        (replace(a, features=a.features.astype(np.float32)), b),
        (replace(a, targets=a.targets.astype(np.float32)), b),
        (replace(a, features=np.full((1, 2), np.nan)), b),
        (replace(a, features=np.broadcast_to(a.features[:, :1], (1, 1_000_001))), b),
        (replace(a, targets=np.array([[2.0]])), b),
        (replace(a, targets=np.ones(1)), b),
        (a, replace(b, features=a.features)),
        (replace(a, targets=a.features[:, :1]), b),
        (
            replace(
                a,
                features=np.ones((1025, 2)),
                targets=np.ones((1025, 1)),
                row_ids=tuple(f"large-{i}" for i in range(1025)),
            ),
            b,
        ),
    ]


@pytest.mark.parametrize("data", invalid_data())
def test_should_reject_constructor_data_before_any_copy_or_callback(data, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid constructor reached gate")

    monkeypatch.setattr(consolidation, "gate_coupled_scaling_prune", forbidden)
    with pytest.raises(ValueError):
        operation(data)


def test_should_refuse_final_role_sentinel_before_inspecting_or_copying_payload():
    class Sentinel:
        def __deepcopy__(self, memo):
            raise AssertionError("final payload copied")

    a, b = probes()
    bad = replace(a, role="final_test", features=Sentinel(), targets=Sentinel())
    with pytest.raises(ValueError):
        operation((bad, b))


@pytest.mark.parametrize("removed", ([], (), (True, 1), (np.int64(1), 1), (-1, 1), (0,) * 17))
def test_should_validate_native_constructor_removed_coordinates(removed: Any):
    with pytest.raises(ValueError):
        CoupledScalingPruneConsolidator(
            removed,
            probes(),
            limits=CoupledPruneLimits(1, 1, 1),
            budget=budget(),
            inference_steps=2,
            inference_learning_rate=0.01,
        )


@pytest.mark.parametrize(
    "kind", ("removed_count", "removed_width", "probe_width", "activation_cap")
)
def test_should_refuse_geometry_dependent_limits_before_gate(kind, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid geometry reached gate")

    monkeypatch.setattr(consolidation, "gate_coupled_scaling_prune", forbidden)
    state = raw()
    removed: tuple[int, ...] = (1, 1)
    data = probes()
    if kind == "removed_count":
        removed = (1,)
    elif kind == "removed_width":
        removed = (3, 1)
    elif kind == "probe_width":
        data = tuple(replace(p, features=np.ones((1, 3))) for p in data)
    else:
        state = raw(width=64, hidden=16)
        removed = (0,) * 16
        data = (
            ScalingBatch(
                "validation",
                tuple(f"many-{i}" for i in range(1024)),
                np.ones((1024, 2)),
                np.ones((1024, 1)),
            ),
        )
    callback = CoupledScalingPruneConsolidator(
        removed,
        data,
        limits=CoupledPruneLimits(1, 1, 1),
        budget=budget(),
        inference_steps=2,
        inference_learning_rate=0.01,
    )
    with pytest.raises(ValueError):
        callback(CoupledScalingTopologyState(1, state))


def test_should_consolidate_real_raw_pc_neutral_active_then_train_only_candidate_with_history():
    results = {}
    for family in ("pc", "neutral", "active"):
        state = raw(family)
        for layer, (w, b) in enumerate(
            zip(
                state.raw_parameters.hidden_weights, state.raw_parameters.hidden_biases, strict=True
            )
        ):
            w[:, 1] = 0
            b[:, 1] = 0
            if layer:
                w[1, :] = 0
        state.raw_parameters.output_weight[1, :] = 0
        rt, clock, session, source = runtime(state)
        before, actor = source.snapshot_state(), rt.actor.snapshot_state()
        receipt = rt.consolidate("prune", operation(limits=CoupledPruneLimits(1e-12, 1e-12, 1e-12)))
        assert (
            receipt.attempt_number == 1
            and receipt.diagnostic.definition == "coupled_scaling_prune_max_probability_change_v1"
        )
        pruned = rt.candidate_snapshot().state
        assert pruned.raw_state.config.width == 2 and len(pruned.prune_lineage) == 1
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
        reference = step_coupled_scaling(pruned.raw_state, x, y, 0.03, 2, 0.05)
        assert len(applied) == 1 and applied[0].diagnostic.value == reference.evaluation.energy
        candidate = rt.candidate_snapshot()
        np.testing.assert_equal(asdict(candidate.state.raw_state), asdict(reference.state))
        assert candidate.state.prune_lineage == pruned.prune_lineage
        assert candidate.wake_updates == session.updates_completed == 1
        np.testing.assert_equal(asdict(rt.actor.snapshot_state()), asdict(actor))
        np.testing.assert_array_equal(rt.actor.predict(x).prediction, prediction)
        np.testing.assert_equal(asdict(source.snapshot_state()), asdict(before))
        results[family] = candidate.state.raw_state
    np.testing.assert_equal(
        asdict(results["pc"].raw_parameters), asdict(results["neutral"].raw_parameters)
    )
