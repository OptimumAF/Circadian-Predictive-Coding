"""Owned topology ports through real consolidation and delayed native wake.

Only the three-variant integration control trains: six native calls, plus twelve
guarded gate inferences. Other training dispatch and gate controls use fakes.
"""

from copy import deepcopy
from dataclasses import asdict, replace
from typing import Any

import numpy as np
import pytest

from src.adapters.coupled_learners import CoupledLearner
from src.adapters.topology_coupled_learner import TopologyCoupledLearner
import src.app.coupled_prune_consolidation as consolidation
from src.app.coupled_prune_consolidation import CoupledPruneConsolidator, CoupledPruneRefused
from src.app.coupled_pruning_gate import (
    CoupledPruneDecision,
    CoupledPruneDiagnostic,
    CoupledPruneLimits,
)
from src.app.actor_shadow import ActorShadowRuntime
from src.app.linear_scaling_measurement import ScalingBatch
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.actor_ports import ConsolidatedState
from src.core.coupled_circadian_predictive_coding import (
    CoupledCircadianConfig,
    CoupledCircadianPredictiveCodingNetwork,
)
from src.core.coupled_inference import CoupledInferenceDiagnostics, CoupledInferencePolicy
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork, CoupledTrainResult
from src.core.coupled_topology import capture_coupled_topology, restore_coupled_topology
from src.core.coupled_width_pruning import copy_coupled_model, prune_coupled_width
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.learner_ports import TrainingDiagnostic


def same(actual, expected):
    np.testing.assert_equal(asdict(actual), asdict(expected))


def native(family="pc"):
    result: CoupledPredictiveCodingNetwork
    if family == "pc":
        result = CoupledPredictiveCodingNetwork(2, (3, 3), seed=23)
    else:
        config = CoupledCircadianConfig(0.9, 0.2, 0.7, 1.0 if family == "neutral" else 0.2)
        result = CoupledCircadianPredictiveCodingNetwork(2, (3, 3), seed=23, config=config)
        state = result.snapshot_state()
        assert state.wake_state is not None
        chemicals = np.array([0.3, 0.7, 1.1])
        gates = np.clip(np.exp(-0.7 * chemicals), config.min_plasticity, 1.0)
        result.restore_state(
            replace(
                state,
                wake_state=replace(
                    state.wake_state,
                    chemical_state=chemicals,
                    plasticity_state=gates,
                    wake_updates=4,
                ),
            )
        )
    # Declare one genuinely disconnected last-hidden coordinate.
    p = result.snapshot_parameters()
    p.hidden_weights[-1][:, 1] = 0
    p.hidden_biases[-1][:, 1] = 0
    p.output_weight[1, :] = 0
    result.restore_parameters(p)
    return result


def learner(initial=None, **changes):
    settings: dict[str, Any] = dict(
        learning_rate=0.03, inference_steps=2, inference_learning_rate=0.1
    )
    settings.update(changes)
    return TopologyCoupledLearner(native() if initial is None else initial, **settings)


def probes():
    return (
        ScalingBatch("train", ("local-train",), np.array([[0.2, -0.3]]), np.ones((1, 1))),
        ScalingBatch("validation", ("local-guard",), np.array([[0.7, 0.4]]), np.zeros((1, 1))),
    )


def budget(updates=0):
    return ToyBudgetSession(ToyExecutionBudget(max_training_updates=updates), lambda: 0.0)


def operation(probe_batches=None, **changes):
    settings = dict(
        limits=CoupledPruneLimits(1e-10, 1e-10, 1e-10),
        budget=budget(),
        inference_steps=2,
        inference_learning_rate=0.01,
    )
    settings.update(changes)
    return CoupledPruneConsolidator(
        1, 1, probes() if probe_batches is None else probe_batches, **settings
    )


def runtime_for(source=None):
    source = learner() if source is None else source
    clock = LogicalClock()
    resource = budget(1)
    runtime = ActorShadowRuntime(
        source,
        actor_version="a0",
        candidate_version="c0",
        clock=clock,
        budget=resource,
        max_experiences=1,
        max_consolidations=3,
    )
    return runtime, clock, resource, source


def fake_diagnostics(batches):
    inferred = CoupledInferenceDiagnostics("max_steps", 2, 2, 3, 3, 2, 0.8, 0.7, 0.1, False)
    return tuple(
        CoupledPruneDiagnostic(
            batch.role,
            batch.row_ids,
            0.5,
            0.6,
            0.5,
            0.6,
            0.1 * (index + 1),
            0.1,
            0.1,
            inferred,
            inferred,
        )
        for index, batch in enumerate(batches)
    )


def test_should_consolidate_real_pc_neutral_and_active_then_continue_without_changing_actor():
    resulting_parameters = {}
    for family in ("pc", "neutral", "active"):
        original_native = native(family)
        native_before = original_native.snapshot_state()
        source = learner(original_native)
        supplied = source.snapshot_state()
        runtime, clock, resource, _ = runtime_for(source)
        actor_before = runtime.actor.snapshot_state()
        features, targets = np.array([[0.3, -0.4], [0.7, 0.2]]), np.array([[0.0], [0.35]])
        prediction_before = runtime.actor.predict(features)
        receipt = runtime.consolidate("prune-1", operation())
        assert receipt.attempt_number == 1
        assert receipt.diagnostic.definition == "coupled_prune_max_probability_change_v1"
        assert 0 <= receipt.diagnostic.value <= 1e-10
        pruned = runtime.candidate_snapshot()
        assert pruned.state.hidden_dims == (3, 2)
        assert len(pruned.state.prune_lineage) == 1
        assert pruned.consolidations_completed == pruned.consolidation_attempts == 1
        expected = restore_coupled_topology(pruned.state)
        runtime.record_experience(
            Experience(
                "new-row",
                "episode",
                0,
                "a0",
                features,
                "train",
                ExperiencePermissions(training=True),
            )
        )
        runtime.record_label(LabelArrival("new-label", "new-row", "episode", 1, "a0", targets))
        assert runtime.train_ready() == ()
        clock.advance_to(1)
        applied = runtime.train_ready(max_updates=1)
        reference = expected.train_epoch(features, targets, 0.03, 2, 0.1)
        assert len(applied) == 1 and applied[0].diagnostic.value == reference.energy
        after = runtime.candidate_snapshot()
        same(after.state.snapshot, expected.snapshot_state())
        assert after.state.prune_lineage == pruned.state.prune_lineage
        assert after.wake_updates == resource.updates_completed == 1
        assert runtime.train_ready() == ()
        same(runtime.actor.snapshot_state(), actor_before)
        np.testing.assert_array_equal(
            runtime.actor.predict(features).prediction, prediction_before.prediction
        )
        same(source.snapshot_state(), supplied)
        same(original_native.snapshot_state(), native_before)
        assert runtime.actor.version == "a0"
        if family != "pc":
            assert after.state.snapshot.wake_state is not None
            assert after.state.snapshot.wake_state.wake_updates == 5
        resulting_parameters[family] = after.state.snapshot.parameters
    ordinary, controlled = resulting_parameters["pc"], resulting_parameters["neutral"]
    same(ordinary, controlled)


def test_should_keep_owned_training_settings_across_fork_builder_restore_and_shorter_history(
    monkeypatch,
):
    calls = []

    def fake_train(self, x, y, learning_rate, steps, inference_rate, *, inference_policy=None):
        calls.append((learning_rate, steps, inference_rate, deepcopy(inference_policy)))
        p = self.snapshot_parameters()
        self.restore_parameters(replace(p, output_bias=p.output_bias + 1))
        return CoupledTrainResult(0.5)

    monkeypatch.setattr(CoupledPredictiveCodingNetwork, "train_epoch", fake_train)
    policy = CoupledInferencePolicy()
    source = learner(inference_policy=policy)
    original = source.snapshot_state()
    change = prune_coupled_width(restore_coupled_topology(original), 1, 1)
    pruned = capture_coupled_topology(change.model, prune_lineage=(change.lineage,))
    built = source.build_from_state(pruned)
    forked = built.fork()
    object.__setattr__(policy, "gradient_tolerance", 99.0)
    x, y = np.ones((1, 2)), np.zeros((1, 1))
    built.train_batch(x, y)
    same(forked.snapshot_state(), pruned)
    same(source.snapshot_state(), original)
    forked.train_batch(x, y)
    assert calls == [
        (0.03, 2, 0.1, CoupledInferencePolicy()),
        (0.03, 2, 0.1, CoupledInferencePolicy()),
    ]
    source.restore_state(pruned)
    assert source.snapshot_state().prune_lineage == (change.lineage,)
    source.restore_state(original)
    same(source.snapshot_state(), original)
    assert source.snapshot_state().prune_lineage == ()


@pytest.mark.parametrize("kind", ("family", "input", "precision", "depth", "geometry", "lineage"))
def test_should_refuse_incompatible_restore_and_builder_without_changing_learner(kind):
    source = learner()
    before = source.snapshot_state()
    if kind == "family":
        wrong = capture_coupled_topology(native("active"))
    elif kind == "input":
        wrong = capture_coupled_topology(CoupledPredictiveCodingNetwork(3, (3, 3), seed=23))
    elif kind == "precision":
        wrong = capture_coupled_topology(
            CoupledPredictiveCodingNetwork(2, (3, 3), seed=23, precisions=(0.7, 0.9))
        )
    elif kind == "depth":
        wrong = capture_coupled_topology(CoupledPredictiveCodingNetwork(2, (3,), seed=23))
    else:
        wrong = deepcopy(before)
        object.__setattr__(
            wrong,
            "hidden_dims" if kind == "geometry" else "prune_lineage",
            (3, 2) if kind == "geometry" else ("bad",),
        )
    with pytest.raises(ValueError):
        source.restore_state(wrong)
    with pytest.raises(ValueError):
        source.build_from_state(wrong)
    same(source.snapshot_state(), before)


def test_should_pin_cpc_policy_and_preserve_existing_strict_adapter_restore():
    initial = native("active")
    source = learner(initial)
    before = source.snapshot_state()
    incompatible = CoupledCircadianPredictiveCodingNetwork(
        2, (3, 3), seed=23, config=CoupledCircadianConfig(0.9, 0.4, 0.7, 0.2)
    )
    with pytest.raises(ValueError):
        source.restore_state(capture_coupled_topology(incompatible))
    same(source.snapshot_state(), before)
    strict = CoupledLearner(
        initial, learning_rate=0.03, inference_steps=2, inference_learning_rate=0.1
    )
    strict_before = strict.snapshot_state()
    pruned = prune_coupled_width(initial, 1, 1)
    with pytest.raises(ValueError):
        strict.restore_state(pruned.model.snapshot_state())
    same(strict.snapshot_state(), strict_before)


def test_should_append_existing_history_and_copy_constructor_probes_and_limits(monkeypatch):
    initial = native("active")
    previous = prune_coupled_width(initial, 0, 1)
    wrapper = capture_coupled_topology(previous.model, prune_lineage=(previous.lineage,))
    saved = deepcopy(wrapper)
    local_probes = probes()
    old_features = tuple(p.features.copy() for p in local_probes)
    limits = CoupledPruneLimits(1.0, 1.0, 1.0)
    callback = operation(local_probes, limits=limits)
    for p in local_probes:
        p.features.fill(99)
        p.targets.fill(99)
    object.__setattr__(limits, "max_prediction_change", 0.0)

    def accepted(model, layer, unit, batches, *, limits, **kwargs):
        for actual, expected in zip(batches, old_features, strict=True):
            np.testing.assert_array_equal(actual.features, expected)
        assert limits.max_prediction_change == 1.0
        changed = prune_coupled_width(model, layer, unit)
        return CoupledPruneDecision(
            "accepted", None, changed.model, changed.lineage, fake_diagnostics(batches), 2
        )

    monkeypatch.setattr(consolidation, "gate_coupled_prune", accepted)
    result = callback(wrapper)
    assert type(result) is ConsolidatedState
    assert result.state.hidden_dims == (2, 2)
    assert result.state.prune_lineage[:-1] == (previous.lineage,)
    assert len(result.state.prune_lineage) == 2
    assert result.diagnostic.definition == "coupled_prune_max_probability_change_v1"
    assert result.diagnostic.value == 0.2
    same(wrapper, saved)


@pytest.mark.parametrize("status", ("rejected", "failed"))
def test_should_charge_refused_attempts_without_publication_or_stopping_candidate(
    monkeypatch, status
):
    decisions = []

    def refuse(initial, layer, unit, batches, **kwargs):
        decision = CoupledPruneDecision(
            status,
            "deliberate refusal",
            copy_coupled_model(initial),
            None,
            fake_diagnostics(batches),
            2,
        )
        decisions.append(decision)
        return decision

    monkeypatch.setattr(consolidation, "gate_coupled_prune", refuse)
    runtime, _, _, source = runtime_for(learner(native("active")))
    before, actor_before = runtime.candidate_snapshot(), runtime.actor.snapshot_state()
    with pytest.raises(CoupledPruneRefused) as refusal:
        runtime.consolidate("refused", operation())
    assert isinstance(refusal.value, ValueError)
    assert refusal.value.decision.status == status
    assert refusal.value.decision.lineage is None
    candidate = runtime.candidate_snapshot()
    same(candidate.state, before.state)
    same(runtime.actor.snapshot_state(), actor_before)
    assert candidate.consolidation_attempts == 1 and candidate.consolidations_completed == 0
    decision_parameters = decisions[0].model.snapshot_parameters()
    decisions[0].model.restore_parameters(
        replace(decision_parameters, output_bias=np.array([[99.0]]))
    )
    assert refusal.value.decision.model.snapshot_parameters().output_bias[0, 0] != 99
    with pytest.raises(ValueError, match="duplicate"):
        runtime.consolidate("refused", operation())
    receipt = runtime.consolidate(
        "next", lambda state: ConsolidatedState(state, TrainingDiagnostic("noop_v1", 0.0))
    )
    assert receipt.attempt_number == 2
    assert runtime.candidate_snapshot().consolidations_completed == 1
    same(source.snapshot_state(), before.state)


def test_should_keep_candidate_and_actor_stable_if_transform_raises(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("deliberate callback fault")

    monkeypatch.setattr(consolidation, "gate_coupled_prune", fail)
    runtime, _, _, _ = runtime_for()
    before, actor_before = runtime.candidate_snapshot(), runtime.actor.snapshot_state()
    with pytest.raises(RuntimeError, match="deliberate"):
        runtime.consolidate("fault", operation())
    after = runtime.candidate_snapshot()
    same(after.state, before.state)
    same(runtime.actor.snapshot_state(), actor_before)
    assert after.consolidation_attempts == 1 and after.consolidations_completed == 0


def test_should_stop_runtime_after_atomic_refusal_of_malformed_replacement(monkeypatch):
    seen = []
    original_restore = TopologyCoupledLearner.restore_state

    def observe_restore(self, state):
        before = self.snapshot_state()
        try:
            original_restore(self, state)
        except ValueError:
            seen.append((before, self.snapshot_state()))
            raise

    monkeypatch.setattr(TopologyCoupledLearner, "restore_state", observe_restore)
    runtime, _, _, source = runtime_for()
    before, actor_before = runtime.candidate_snapshot(), runtime.actor.snapshot_state()

    def malformed(state):
        object.__setattr__(state, "hidden_dims", (3, 2))
        return ConsolidatedState(state, TrainingDiagnostic("invalid_v1", 0.0))

    with pytest.raises(ValueError):
        runtime.consolidate("malformed", malformed)
    assert seen
    same(seen[-1][0], seen[-1][1])
    same(source.snapshot_state(), before.state)
    same(runtime.actor.snapshot_state(), actor_before)
    cursor = runtime.capture_consolidation_cursor()
    assert cursor.stopped is True
    assert runtime.applied_consolidations == ()
    with pytest.raises(ValueError, match="stopped"):
        runtime.candidate_snapshot()


def test_should_preserve_managed_retention_refusal_before_transform_or_attempt():
    runtime, _, _, _ = runtime_for()
    before = runtime.candidate_snapshot()

    class Managed:
        def _require_access(self):
            pass

    runtime.actor._payload_registry._lifecycle = Managed()
    invoked = []

    def forbidden(state):
        invoked.append(True)
        return ConsolidatedState(state, TrainingDiagnostic("forbidden_v1", 0.0))

    with pytest.raises(ValueError, match="managed retention"):
        runtime.consolidate("managed", forbidden)
    assert invoked == []
    after = runtime.candidate_snapshot()
    same(after.state, before.state)
    assert after.consolidation_attempts == after.consolidations_completed == 0
