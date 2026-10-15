"""Raw prune decisions preserve owned fallback state and complete probe evidence."""

from dataclasses import asdict, replace
from typing import Any

import numpy as np
import pytest

import src.app.coupled_scaling_pruning_gate as gate
from src.app.coupled_pruning_gate import CoupledPruneLimits, _Score
from src.app.coupled_scaling_pruning_gate import gate_coupled_scaling_prune
from src.app.linear_scaling_measurement import ScalingBatch
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.coupled_circadian_predictive_coding import CoupledCircadianConfig
from src.core.coupled_inference import CoupledInferenceDiagnostics
from src.core.coupled_pc_scaling import (
    CoupledScalingConfig,
    CoupledScalingState,
    initialize_coupled_scaling,
)
from src.core.coupled_scaling_pruning import prune_coupled_scaling_width


def arrays(p):
    return (*p.hidden_weights, *p.hidden_biases, p.output_weight, p.output_bias)


def initial(family="pc"):
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
        CoupledScalingConfig("mean_field", 2, 3, 2, 2.0, precisions=(0.7, 1.3), circadian=policy),
        41,
    )
    if state.wake_state is not None:
        chemicals = np.array([0.3, 0.7, 1.1])
        assert policy is not None
        wake = replace(
            state.wake_state,
            chemical_state=chemicals,
            plasticity_state=np.clip(
                np.exp(-policy.plasticity_sensitivity * chemicals), policy.min_plasticity, 1.0
            ),
            wake_updates=4,
        )
        state = CoupledScalingState(state.config, state.raw_parameters, wake)
    return state


def probes():
    return (
        ScalingBatch("train", ("train-0",), np.array([[0.2, -0.3]]), np.array([[0.2]])),
        ScalingBatch("validation", ("guard-0",), np.array([[0.7, 0.4]]), np.array([[0.8]])),
    )


def budget():
    return ToyBudgetSession(ToyExecutionBudget(max_training_updates=0), lambda: 0.0)


def run(state=None, probe_batches=None, **options):
    return gate_coupled_scaling_prune(
        initial() if state is None else state,
        options.pop("removed_units", (0, 2)),
        probes() if probe_batches is None else probe_batches,
        limits=options.pop("limits", CoupledPruneLimits(1.0, 2.0, 1.0)),
        budget=options.pop("budget", budget()),
        inference_steps=options.pop("inference_steps", 2),
        inference_learning_rate=options.pop("inference_learning_rate", 0.01),
        **options,
    )


class FakeScores:
    def __init__(self, monkeypatch):
        self.calls = []
        self.mutate = False
        self.fail_at = None
        self.last = None
        self.override = None
        monkeypatch.setattr(gate, "_score", self.score)

    def score(self, model, probe, steps, rate):
        snapshot = model.snapshot_state()
        self.calls.append((snapshot, probe.features.copy(), probe.targets.copy(), steps, rate))
        candidate = model.hidden_dims == (2, 2)
        probability, loss, energy = (0.75, 0.75, 1.25) if candidate else (0.25, 0.5, 1.0)
        inference = CoupledInferenceDiagnostics(
            "max_steps", steps, steps, steps + 1, steps + 1, steps, energy + 0.1, energy, 0.1, False
        )
        result = _Score(np.full((len(probe.features), 1), probability), loss, energy, inference)
        if self.override:
            result = self.override(result)
        self.last = result
        if self.mutate:
            p = model.snapshot_parameters()
            for a in arrays(p):
                a[:] = 999
            model.restore_parameters(p)
            probe.features[:] = 998
            probe.targets[:] = 997
        if len(self.calls) == self.fail_at:
            raise FloatingPointError("score fault")
        return result


@pytest.mark.parametrize(
    "name,threshold,status",
    (
        ("prediction", 0.49, "rejected"),
        ("prediction", 0.5, "accepted"),
        ("bce", 0.24, "rejected"),
        ("bce", 0.25, "accepted"),
        ("energy", 0.24, "rejected"),
        ("energy", 0.25, "accepted"),
    ),
)
def test_should_complete_all_probes_before_each_inclusive_damage_decision(
    monkeypatch, name, threshold, status
):
    fake = FakeScores(monkeypatch)
    settings = dict(
        max_prediction_change=1.0, max_bce_increase=2.0, max_inferred_energy_increase=1.0
    )
    settings[
        {
            "prediction": "max_prediction_change",
            "bce": "max_bce_increase",
            "energy": "max_inferred_energy_increase",
        }[name]
    ] = threshold
    source = initial("active")
    before = asdict(source)
    decision = run(source, limits=CoupledPruneLimits(**settings))
    assert decision.status == status
    assert len(decision.diagnostics) == decision.planned_probe_count == 2
    assert len(fake.calls) == 4
    assert (decision.lineage is not None) == (status == "accepted")
    assert (decision.max_effective_relative_error is not None) == (status == "accepted")
    assert decision.state.config.width == (2 if status == "accepted" else 3)
    for diagnostic, batch in zip(decision.diagnostics, probes(), strict=True):
        assert diagnostic.role == batch.role and diagnostic.row_ids == batch.row_ids
        assert diagnostic.max_prediction_change == 0.5
        assert diagnostic.bce_increase == diagnostic.inferred_energy_increase == 0.25
        assert (
            diagnostic.source_inference.accepted_steps
            == diagnostic.candidate_inference.accepted_steps
            == 2
        )
    np.testing.assert_equal(asdict(source), before)
    if status == "rejected":
        np.testing.assert_equal(asdict(decision.state), before)


def test_should_give_every_score_private_models_buffers_and_copy_returned_scores_before_callbacks(
    monkeypatch,
):
    source = initial("active")
    before = asdict(source)
    fake = FakeScores(monkeypatch)
    fake.mutate = True
    session = budget()

    def mutate_retained_result():
        if fake.last is not None:
            fake.last.probabilities[:] = 999

    monkeypatch.setattr(session, "before_final", mutate_retained_result)
    decision = run(source, budget=session)
    assert decision.status == "accepted"
    assert len(fake.calls) == 4
    for first, again in (
        (fake.calls[0][0], fake.calls[2][0]),
        (fake.calls[1][0], fake.calls[3][0]),
    ):
        np.testing.assert_equal(asdict(first), asdict(again))
    for index, call in enumerate(fake.calls):
        np.testing.assert_array_equal(call[1], probes()[index // 2].features)
        np.testing.assert_array_equal(call[2], probes()[index // 2].targets)
    assert all(d.max_prediction_change == 0.5 for d in decision.diagnostics)
    np.testing.assert_equal(asdict(source), before)


@pytest.mark.parametrize("fail_at", (1, 2, 3, 4))
def test_should_keep_private_full_raw_fallback_and_only_completed_probe_diagnostics_on_score_fault(
    monkeypatch, fail_at
):
    source = initial("active")
    before = asdict(source)
    fake = FakeScores(monkeypatch)
    fake.mutate = True
    fake.fail_at = fail_at
    decision = run(source)
    assert decision.status == "failed" and "score fault" in decision.reason
    assert decision.lineage is decision.max_effective_relative_error is None
    assert len(decision.diagnostics) == (fail_at - 1) // 2
    np.testing.assert_equal(asdict(decision.state), before)
    np.testing.assert_equal(asdict(source), before)
    decision.state.raw_parameters.output_bias[:] = 999
    assert decision.state.wake_state is not None
    decision.state.wake_state.chemical_state[:] = 999
    np.testing.assert_equal(asdict(source), before)


def test_should_own_source_probes_limits_and_lineage_before_budget_callback(monkeypatch):
    source = initial("active")
    data = probes()
    before = asdict(source)
    limits = CoupledPruneLimits(1.0, 2.0, 1.0)
    fake = FakeScores(monkeypatch)
    session = budget()

    def mutate():
        source.raw_parameters.output_bias[:] = 99
        assert source.wake_state is not None
        source.wake_state.chemical_state[:] = 99
        for p in data:
            p.features[:] = 99
            p.targets[:] = 99
        object.__setattr__(limits, "max_prediction_change", 0.0)

    monkeypatch.setattr(session, "before_final", mutate)
    decision = run(source, data, limits=limits, budget=session)
    assert decision.status == "accepted"
    for index, call in enumerate(fake.calls):
        np.testing.assert_array_equal(call[1], probes()[index // 2].features)
        np.testing.assert_array_equal(call[2], probes()[index // 2].targets)
    assert decision.lineage is not None
    np.testing.assert_equal(asdict(decision.lineage.source_config), before["config"])
    assert decision.state.raw_parameters.output_bias[0, 0] == 0


def test_should_refuse_candidate_authority_on_final_budget_failure_with_all_probe_evidence(
    monkeypatch,
):
    fake = FakeScores(monkeypatch)
    source = initial()
    before = asdict(source)
    session = budget()
    ready = [False]

    def final_fault():
        if len(fake.calls) == 4:
            if ready[0]:
                raise RuntimeError("late budget fault")
            ready[0] = True

    monkeypatch.setattr(session, "before_final", final_fault)
    decision = run(source, budget=session)
    assert decision.status == "failed" and "late budget fault" in decision.reason
    assert len(decision.diagnostics) == 2
    assert decision.lineage is decision.max_effective_relative_error is None
    np.testing.assert_equal(asdict(decision.state), before)


@pytest.mark.parametrize("malformed", ("family", "lineage", "geometry", "error", "wake"))
def test_should_refuse_malformed_candidate_transfer_without_scoring_or_losing_raw_fallback(
    monkeypatch, malformed
):
    source = initial("active")
    before = asdict(source)
    fake = FakeScores(monkeypatch)

    def bad_candidate(state, removed):
        result = prune_coupled_scaling_width(state, removed)
        if malformed == "family":
            return asdict(result)
        if malformed == "lineage":
            object.__setattr__(result.lineage, "removed_units", (1, 1))
        elif malformed == "geometry":
            object.__setattr__(result.state.config, "width", 3)
        elif malformed == "error":
            object.__setattr__(result, "max_effective_relative_error", 1.0)
        else:
            assert result.state.wake_state is not None
            object.__setattr__(result.state.wake_state, "wake_updates", 0)
        return result

    monkeypatch.setattr(gate, "prune_coupled_scaling_width", bad_candidate)
    decision = run(source)
    assert (
        decision.status == "failed"
        and decision.lineage is decision.max_effective_relative_error is None
    )
    assert fake.calls == []
    np.testing.assert_equal(asdict(decision.state), before)


@pytest.mark.parametrize("bad", ("early", "fallback", "nan", "dtype", "shape", "range"))
def test_should_refuse_invalid_score_records_instead_of_accepting_partial_inference(
    monkeypatch, bad
):
    fake = FakeScores(monkeypatch)

    def override(result):
        if bad == "early":
            return replace(
                result,
                inference=replace(result.inference, reason="gradient_tolerance", accepted_steps=1),
            )
        if bad == "fallback":
            return replace(result, inference=replace(result.inference, used_fallback=True))
        if bad == "nan":
            return replace(result, energy=float("nan"))
        if bad == "dtype":
            return replace(result, probabilities=result.probabilities.astype(np.float32))
        if bad == "shape":
            return replace(result, probabilities=result.probabilities.ravel())
        return replace(result, probabilities=np.full_like(result.probabilities, 1.1))

    fake.override = override
    decision = run()
    assert (
        decision.status == "failed"
        and decision.lineage is decision.max_effective_relative_error is None
    )


def invalid_probes():
    a, b = probes()
    return [
        (),
        [a, b],
        (a, b) * 5,
        (a,),
        (replace(a, role="final_test"), b),
        (a, replace(b, row_ids=a.row_ids)),
        (replace(a, row_ids=(" padded ",)), b),
        (replace(a, row_ids=("line\nbreak",)), b),
        (replace(a, features=a.features.astype(np.float32)), b),
        (replace(a, targets=np.array([[1.1]])), b),
        (replace(a, features=np.ones((1, 3))), b),
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


@pytest.mark.parametrize("probe_batches", invalid_probes())
def test_should_preflight_roles_ids_storage_and_bounds_before_any_external_callback(
    monkeypatch, probe_batches
):
    source = initial()
    fake = FakeScores(monkeypatch)
    session = budget()
    calls = []
    monkeypatch.setattr(session, "before_final", lambda: calls.append("budget"))
    with pytest.raises(ValueError):
        run(source, probe_batches, budget=session)
    assert calls == [] and fake.calls == []


@pytest.mark.parametrize(
    "options",
    (
        {"removed_units": [0, 1]},
        {"removed_units": (True, 1)},
        {"limits": {}},
        {"budget": {}},
        {"inference_steps": True},
        {"inference_steps": 0},
        {"inference_steps": 65},
        {"inference_learning_rate": np.float64(0.01)},
        {"inference_learning_rate": 0},
    ),
)
def test_should_preflight_native_controls_without_scores(monkeypatch, options: Any):
    fake = FakeScores(monkeypatch)
    with pytest.raises(ValueError):
        run(**options)
    assert fake.calls == []


@pytest.mark.parametrize("family", ("pc", "neutral", "active"))
def test_should_accept_real_disconnected_uniform_prune_and_keep_source_chemistry_unchanged(family):
    source = initial(family)
    p = source.raw_parameters
    for layer, (w, b) in enumerate(zip(p.hidden_weights, p.hidden_biases, strict=True)):
        w[:, 1] = 0
        b[:, 1] = 0
        if layer:
            w[1, :] = 0
    p.output_weight[1, :] = 0
    before = asdict(source)
    decision = run(source, removed_units=(1, 1), limits=CoupledPruneLimits(1e-12, 1e-12, 1e-12))
    assert decision.status == "accepted" and decision.state.config.width == 2
    assert decision.lineage is not None and decision.max_effective_relative_error is not None
    assert len(decision.diagnostics) == 2
    for diagnostic in decision.diagnostics:
        assert diagnostic.max_prediction_change <= 1e-12
        assert diagnostic.bce_increase <= 1e-12 and diagnostic.inferred_energy_increase <= 1e-12
        assert (
            diagnostic.source_inference.accepted_steps
            == diagnostic.candidate_inference.accepted_steps
            == 2
        )
    np.testing.assert_equal(asdict(source), before)
    if source.wake_state is not None:
        assert decision.state.wake_state is not None
        assert decision.state.wake_state.wake_updates == 4
        np.testing.assert_array_equal(
            decision.state.wake_state.chemical_state, source.wake_state.chemical_state[[0, 2]]
        )
