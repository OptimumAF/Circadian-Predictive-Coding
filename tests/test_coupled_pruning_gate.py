"""Independent bounded nonlinear prune probes and private-fallback isolation."""

from dataclasses import asdict, replace
from typing import Any

import numpy as np
import pytest

import src.app.coupled_pruning_gate as gate
from src.app.coupled_pruning_gate import CoupledPruneLimits, gate_coupled_prune
from src.app.linear_scaling_measurement import ScalingBatch
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.coupled_circadian_predictive_coding import (
    CoupledCircadianConfig,
    CoupledCircadianPredictiveCodingNetwork,
)
from src.core.coupled_energy import CoupledEvaluation, CoupledParameters
from src.core.coupled_inference import CoupledInferenceDiagnostics, CoupledInferenceResult
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork
from src.core.coupled_width_pruning import prune_coupled_width


def arrays(p):
    return (*p.hidden_weights, *p.hidden_biases, p.output_weight, p.output_bias)


def model(family="pc"):
    result: CoupledPredictiveCodingNetwork
    if family == "pc":
        result = CoupledPredictiveCodingNetwork(2, (3, 3), seed=23)
    else:
        config = CoupledCircadianConfig(0.9, 0.2, 0.7, 1.0 if family == "neutral" else 0.2)
        result = CoupledCircadianPredictiveCodingNetwork(2, (3, 3), seed=23, config=config)
        snapshot = result.snapshot_state()
        assert snapshot.wake_state is not None
        chemical = np.array([0.3, 0.7, 1.1])
        gates = np.clip(
            np.exp(-config.plasticity_sensitivity * chemical), config.min_plasticity, 1.0
        )
        result.restore_state(
            replace(
                snapshot,
                wake_state=replace(
                    snapshot.wake_state,
                    chemical_state=chemical,
                    plasticity_state=gates,
                    wake_updates=4,
                ),
            )
        )
    return result


def probes():
    return (
        ScalingBatch("train", ("train",), np.array([[0.2, -0.3]]), np.ones((1, 1))),
        ScalingBatch("validation", ("guard",), np.array([[0.7, 0.4]]), np.zeros((1, 1))),
    )


def budget():
    return ToyBudgetSession(ToyExecutionBudget(max_training_updates=0), lambda: 0.0)


def run(initial=None, probe_batches=None, **options):
    return gate_coupled_prune(
        model() if initial is None else initial,
        options.pop("layer_index", 1),
        options.pop("unit_index", 1),
        probes() if probe_batches is None else probe_batches,
        limits=options.pop("limits", CoupledPruneLimits(1.0, 1.0, 1.0)),
        budget=options.pop("budget", budget()),
        inference_steps=options.pop("inference_steps", 2),
        inference_learning_rate=options.pop("inference_learning_rate", 0.01),
        **options,
    )


class FakeScores:
    def __init__(self, monkeypatch):
        self.predictions, self.inferences = [], []
        self.mutate = False
        self.fail_prediction = None
        self.fail_inference = None
        self.bad_inference = None
        self.prediction_override = None
        self.last_prediction = None
        self.last_result = None
        monkeypatch.setattr(gate, "predict_coupled", self.predict)
        monkeypatch.setattr(gate, "infer_coupled", self.infer)

    def predict(self, p, x):
        self.predictions.append((p, tuple(w.copy() for w in arrays(p)), x.copy()))
        probability = 0.4 if p.output_weight.shape[0] == 3 else 0.6
        if self.mutate:
            for a in (*arrays(p), x):
                a.fill(999)
        if len(self.predictions) == self.fail_prediction:
            raise FloatingPointError("deliberate prediction failure")
        result = (
            np.full((len(x), 1), probability)
            if self.prediction_override is None
            else self.prediction_override
        )
        self.last_prediction = result
        return result

    def infer(self, p, x, y, precisions, *, max_steps, learning_rate, policy):
        self.inferences.append(
            (p, tuple(w.copy() for w in arrays(p)), x.copy(), y.copy(), policy, max_steps)
        )
        energy = 0.5 if p.output_weight.shape[0] == 3 else 0.7
        latents = tuple(np.zeros((len(x), w.shape[1])) for w in p.hidden_weights)
        gradient = CoupledParameters(
            tuple(np.zeros_like(w) for w in p.hidden_weights),
            tuple(np.zeros_like(b) for b in p.hidden_biases),
            np.zeros_like(p.output_weight),
            np.zeros_like(p.output_bias),
        )
        evaluation = CoupledEvaluation(
            energy, energy, (0.0,) * len(latents), tuple(z.copy() for z in latents), gradient
        )
        diagnostic = CoupledInferenceDiagnostics(
            "max_steps",
            max_steps,
            max_steps,
            max_steps + 1,
            max_steps + 1,
            max_steps,
            energy + 0.1,
            energy,
            0.1,
            False,
        )
        if self.bad_inference == "early":
            diagnostic = replace(
                diagnostic, reason="gradient_tolerance", attempted_steps=1, accepted_steps=1
            )
        elif self.bad_inference == "fallback":
            diagnostic = replace(
                diagnostic,
                reason="energy_increase",
                accepted_steps=max_steps - 1,
                used_fallback=True,
            )
        elif self.bad_inference == "incomplete":
            diagnostic = replace(diagnostic, accepted_steps=max_steps - 1)
        result = CoupledInferenceResult(latents, evaluation, diagnostic)
        self.last_result = result
        if self.mutate:
            for a in (*arrays(p), x, y):
                a.fill(999)
        if len(self.inferences) == self.fail_inference:
            raise FloatingPointError("deliberate inference failure")
        return result


@pytest.mark.parametrize("limit,status", ((0.19, "rejected"), (0.2, "accepted")))
def test_should_finish_every_probe_before_inclusive_observed_damage_decision(
    monkeypatch, limit, status
):
    fake = FakeScores(monkeypatch)
    initial = model("cpc")
    before = initial.snapshot_state()
    decision = run(initial, limits=CoupledPruneLimits(limit, 1.0, 1.0))
    assert decision.status == status
    assert len(decision.diagnostics) == decision.planned_probe_count == 2
    assert len(fake.predictions) == len(fake.inferences) == 4
    assert (decision.lineage is not None) == (status == "accepted")
    assert decision.model.hidden_dims == ((3, 2) if status == "accepted" else (3, 3))
    for index, diagnostic in enumerate(decision.diagnostics):
        assert diagnostic.role == probes()[index].role
        assert diagnostic.row_ids == probes()[index].row_ids
        assert diagnostic.max_prediction_change == pytest.approx(0.2)
        assert (
            diagnostic.source_inferred_energy == 0.5 and diagnostic.candidate_inferred_energy == 0.7
        )
        assert diagnostic.inferred_energy_increase == pytest.approx(0.2)
        y = float(probes()[index].targets[0, 0])
        source_bce = -y * np.log(0.4) - (1 - y) * np.log(0.6)
        candidate_bce = -y * np.log(0.6) - (1 - y) * np.log(0.4)
        assert diagnostic.source_bce == pytest.approx(source_bce)
        assert diagnostic.candidate_bce == pytest.approx(candidate_bce)
        assert diagnostic.bce_increase == pytest.approx(max(0, candidate_bce - source_bce))
        assert (
            diagnostic.source_inference.reason
            == diagnostic.candidate_inference.reason
            == "max_steps"
        )
    for *_, policy, cap in fake.inferences:
        assert policy.min_steps == cap == 2
        assert policy.gradient_tolerance is policy.relative_energy_tolerance is None
    np.testing.assert_equal(asdict(initial.snapshot_state()), asdict(before))
    if status == "rejected":
        np.testing.assert_equal(asdict(decision.model.snapshot_state()), asdict(before))


@pytest.mark.parametrize("family", ("pc", "neutral", "cpc"))
def test_should_accept_real_disconnected_channel_with_full_guarded_inference(family):
    initial = model(family)
    p = initial.snapshot_parameters()
    weights, biases, output = list(p.hidden_weights), list(p.hidden_biases), p.output_weight.copy()
    weights[-1][:, 1] = 0
    biases[-1][:, 1] = 0
    output[1, :] = 0
    initial.restore_parameters(
        replace(p, hidden_weights=tuple(weights), hidden_biases=tuple(biases), output_weight=output)
    )
    before = initial.snapshot_state()
    decision = run(initial, limits=CoupledPruneLimits(1e-10, 1e-10, 1e-10))
    assert decision.status == "accepted"
    assert decision.model.hidden_dims == (3, 2)
    assert len(decision.diagnostics) == 2
    for diagnostic in decision.diagnostics:
        assert (
            diagnostic.source_inference.attempted_steps
            == diagnostic.source_inference.accepted_steps
            == 2
        )
        assert (
            diagnostic.candidate_inference.attempted_steps
            == diagnostic.candidate_inference.accepted_steps
            == 2
        )
        assert (
            not diagnostic.source_inference.used_fallback
            and not diagnostic.candidate_inference.used_fallback
        )
    np.testing.assert_equal(asdict(initial.snapshot_state()), asdict(before))
    if type(initial) is CoupledCircadianPredictiveCodingNetwork:
        assert type(decision.model) is CoupledCircadianPredictiveCodingNetwork
        assert decision.model.wake_updates == 4


def test_should_observe_real_relaxed_energy_damage_even_when_predictions_and_bce_match():
    initial = CoupledPredictiveCodingNetwork(1, (2,), seed=23)
    initial.restore_parameters(
        CoupledParameters(
            (np.zeros((1, 2)),), (np.zeros((1, 2)),), np.array([[0.0], [2.0]]), np.zeros((1, 1))
        )
    )
    probe = ScalingBatch("validation", ("guard",), np.zeros((1, 1)), np.ones((1, 1)))
    decision = run(
        initial,
        (probe,),
        layer_index=0,
        inference_learning_rate=0.1,
        limits=CoupledPruneLimits(0.0, 0.0, 0.0),
    )
    assert decision.status == "rejected"
    assert len(decision.diagnostics) == 1
    diagnostic = decision.diagnostics[0]
    assert diagnostic.max_prediction_change == diagnostic.bce_increase == 0
    assert diagnostic.inferred_energy_increase > 0.05
    assert (
        diagnostic.source_inference.reason == diagnostic.candidate_inference.reason == "max_steps"
    )


def test_should_isolate_predict_and_infer_models_buffers_and_private_fallback(monkeypatch):
    fake = FakeScores(monkeypatch)
    fake.mutate = True
    initial, batches = model("cpc"), probes()
    before = initial.snapshot_state()
    old_probes = tuple((p.features.copy(), p.targets.copy()) for p in batches)
    decision = run(initial, batches, limits=CoupledPruneLimits(0.19, 1.0, 1.0))
    assert decision.status == "rejected"
    for index, (_, saved_arrays, x) in enumerate(fake.predictions):
        assert all(np.max(np.abs(a)) < 10 for a in saved_arrays)
        np.testing.assert_array_equal(x, old_probes[index // 2][0])
    for index, (_, saved_arrays, x, y, _, _) in enumerate(fake.inferences):
        assert all(np.max(np.abs(a)) < 10 for a in saved_arrays)
        np.testing.assert_array_equal(x, old_probes[index // 2][0])
        np.testing.assert_array_equal(y, old_probes[index // 2][1])
    np.testing.assert_equal(asdict(initial.snapshot_state()), asdict(before))
    np.testing.assert_equal(asdict(decision.model.snapshot_state()), asdict(before))
    for actual, expected in zip(batches, old_probes, strict=True):
        np.testing.assert_array_equal(actual.features, expected[0])
        np.testing.assert_array_equal(actual.targets, expected[1])


@pytest.mark.parametrize("stage", ("prediction", "inference", "late_inference"))
def test_should_keep_owned_source_and_only_completed_diagnostics_on_mutating_fault(
    monkeypatch, stage
):
    fake = FakeScores(monkeypatch)
    fake.mutate = True
    if stage == "prediction":
        fake.fail_prediction = 1
    else:
        fake.fail_inference = 4 if stage == "late_inference" else 1
    initial = model("cpc")
    before = initial.snapshot_state()
    decision = run(initial)
    assert decision.status == "failed" and decision.lineage is None
    assert "deliberate" in decision.reason
    assert len(decision.diagnostics) == (1 if stage == "late_inference" else 0)
    assert decision.planned_probe_count == 2
    np.testing.assert_equal(asdict(decision.model.snapshot_state()), asdict(before))
    np.testing.assert_equal(asdict(initial.snapshot_state()), asdict(before))


@pytest.mark.parametrize("mode", ("early", "fallback", "incomplete"))
def test_should_refuse_truncated_or_fallback_inference_instead_of_accepting_damage(
    monkeypatch, mode
):
    fake = FakeScores(monkeypatch)
    fake.bad_inference = mode
    decision = run()
    assert decision.status == "failed" and decision.lineage is None
    assert decision.diagnostics == () and decision.planned_probe_count == 2
    assert len(fake.inferences) == 1


@pytest.mark.parametrize(
    "prediction",
    (
        [[0.5]],
        np.ones((1,)),
        np.ones((1, 1), dtype=np.float32),
        np.full((1, 1), np.nan),
        np.full((1, 1), -0.1),
        np.full((1, 1), 1.1),
    ),
)
def test_should_fail_on_invalid_prediction_without_partial_publication(monkeypatch, prediction):
    fake = FakeScores(monkeypatch)
    fake.prediction_override = prediction
    decision = run()
    assert decision.status == "failed" and decision.lineage is None
    assert decision.diagnostics == ()
    assert fake.inferences == []


def test_should_detach_prediction_energy_and_inference_outputs_before_budget_callbacks(monkeypatch):
    fake = FakeScores(monkeypatch)
    owned_budget = budget()
    original_check = owned_budget.before_final

    def callback():
        original_check()
        if fake.last_prediction is not None:
            fake.last_prediction.fill(99)
        if fake.last_result is not None:
            object.__setattr__(fake.last_result.evaluation, "energy", 99.0)
            object.__setattr__(fake.last_result.diagnostics, "used_fallback", True)

    monkeypatch.setattr(owned_budget, "before_final", callback)
    decision = run(budget=owned_budget)
    assert decision.status == "accepted"
    for d in decision.diagnostics:
        assert d.max_prediction_change == pytest.approx(0.2)
        assert d.source_inferred_energy == 0.5 and d.candidate_inferred_energy == 0.7
        assert not d.source_inference.used_fallback and not d.candidate_inference.used_fallback


def test_should_copy_caller_model_policy_limits_and_probes_before_budget_callbacks(monkeypatch):
    fake = FakeScores(monkeypatch)
    initial, batches = model("cpc"), probes()
    limits = CoupledPruneLimits(1.0, 1.0, 1.0)
    original_probes = tuple((p.features.copy(), p.targets.copy()) for p in batches)
    owned_budget = budget()
    original_check = owned_budget.before_final
    fired = [False]

    def callback():
        original_check()
        if not fired[0]:
            fired[0] = True
            object.__setattr__(initial.config, "chemical_buildup_rate", 99.0)
            initial.restore_parameters(
                replace(initial.snapshot_parameters(), output_bias=np.array([[99.0]]))
            )
            object.__setattr__(limits, "max_prediction_change", 0.0)
            for p in batches:
                p.features.fill(99)
                p.targets.fill(99)

    monkeypatch.setattr(owned_budget, "before_final", callback)
    decision = run(initial, batches, limits=limits, budget=owned_budget)
    assert fired[0] and decision.status == "accepted"
    assert type(decision.model) is CoupledCircadianPredictiveCodingNetwork
    assert decision.model.config.chemical_buildup_rate == 0.2
    for index, (_, saved_arrays, x, y, _, _) in enumerate(fake.inferences):
        assert all(np.max(np.abs(a)) < 10 for a in saved_arrays)
        np.testing.assert_array_equal(x, original_probes[index // 2][0])
        np.testing.assert_array_equal(y, original_probes[index // 2][1])


def test_should_fail_if_a_valid_native_candidate_restarts_chemistry(monkeypatch):
    fake = FakeScores(monkeypatch)

    def reset(initial, layer, unit):
        change = prune_coupled_width(initial, layer, unit)
        snapshot = change.model.snapshot_state()
        assert snapshot.wake_state is not None
        wake = snapshot.wake_state
        change.model.restore_state(
            replace(
                snapshot,
                wake_state=replace(
                    wake,
                    chemical_state=np.zeros_like(wake.chemical_state),
                    plasticity_state=np.ones_like(wake.plasticity_state),
                    wake_updates=0,
                ),
            )
        )
        return change

    monkeypatch.setattr(gate, "prune_coupled_width", reset)
    initial = model("cpc")
    original = initial.snapshot_state()
    decision = run(initial)
    assert decision.status == "failed" and decision.lineage is None
    assert fake.predictions == fake.inferences == []
    np.testing.assert_equal(asdict(decision.model.snapshot_state()), asdict(original))


def test_should_retain_complete_diagnostics_on_final_budget_failure(monkeypatch):
    fake = FakeScores(monkeypatch)
    owned_budget = budget()
    original_check = owned_budget.before_final
    post_last = [0]

    def callback():
        original_check()
        if len(fake.inferences) == 4:
            post_last[0] += 1
            if post_last[0] == 2:
                raise RuntimeError("deliberate final budget failure")

    monkeypatch.setattr(owned_budget, "before_final", callback)
    initial = model("cpc")
    original = initial.snapshot_state()
    decision = run(initial, budget=owned_budget)
    assert decision.status == "failed" and decision.lineage is None
    assert len(decision.diagnostics) == decision.planned_probe_count == 2
    np.testing.assert_equal(asdict(decision.model.snapshot_state()), asdict(original))


def invalid_probes():
    t, v = probes()
    bad: Any = v
    return [
        [],
        (),
        (t,),
        (replace(v, role="final_test"),),
        (replace(v, role="outer_selection"),),
        (replace(v, row_ids=(" padded ",)),),
        (replace(v, row_ids=("row\x00id",)),),
        (replace(bad, row_ids=["guard"]),),
        (t, replace(v, row_ids=t.row_ids)),
        (t, replace(v, features=t.features)),
        (t, replace(v, targets=t.targets)),
        (replace(v, targets=v.features[:, :1]),),
        (replace(v, features=v.features.astype(np.float32)),),
        (replace(v, targets=np.array([[1.1]])),),
        (replace(v, targets=np.array([[-0.1]])),),
        (replace(v, targets=np.array([[np.nan]])),),
        (replace(v, features=np.ones((1, 3))),),
        (
            replace(
                v,
                features=np.ones((1025, 2)),
                targets=np.zeros((1025, 1)),
                row_ids=tuple(f"row-{i}" for i in range(1025)),
            ),
        ),
        tuple(
            replace(
                v, row_ids=(f"guard-{i}",), features=v.features.copy(), targets=v.targets.copy()
            )
            for i in range(9)
        ),
    ]


@pytest.mark.parametrize("batches", invalid_probes())
def test_should_reject_invalid_or_final_role_data_before_budget_callbacks(monkeypatch, batches):
    fake = FakeScores(monkeypatch)
    owned_budget = budget()

    def callback():
        raise AssertionError("invalid probe reached budget")

    monkeypatch.setattr(owned_budget, "before_final", callback)
    with pytest.raises(ValueError):
        run(probe_batches=batches, budget=owned_budget)
    assert fake.predictions == fake.inferences == []


@pytest.mark.parametrize("field", (0, 1, 2))
@pytest.mark.parametrize(
    "value", (True, -1.0, float("nan"), float("inf"), np.float64(0.0), 10**1000)
)
def test_should_require_native_finite_nonnegative_limits(field, value: Any):
    values = [0.0, 0.0, 0.0]
    values[field] = value
    with pytest.raises(ValueError):
        CoupledPruneLimits(*values)


@pytest.mark.parametrize(
    "options",
    (
        {"inference_steps": True},
        {"inference_steps": 0},
        {"inference_steps": 65},
        {"inference_steps": np.int64(2)},
        {"inference_learning_rate": float("inf")},
        {"inference_learning_rate": 0},
        {"inference_learning_rate": np.float64(0.01)},
        {"budget": None},
        {"limits": {}},
        {"layer_index": True},
        {"unit_index": -1},
    ),
)
def test_should_reject_invalid_gate_controls_before_scoring(monkeypatch, options: Any):
    fake = FakeScores(monkeypatch)
    with pytest.raises(ValueError):
        run(**options)
    assert fake.predictions == fake.inferences == []


def test_should_check_probe_activation_cap_before_budget_callbacks(monkeypatch):
    fake = FakeScores(monkeypatch)
    initial = CoupledPredictiveCodingNetwork(2, (500, 500), seed=23)
    probe = ScalingBatch(
        "validation",
        tuple(f"guard-{i}" for i in range(1000)),
        np.zeros((1000, 2)),
        np.zeros((1000, 1)),
    )
    owned_budget = budget()

    def callback():
        raise AssertionError("oversized probe reached budget")

    monkeypatch.setattr(owned_budget, "before_final", callback)
    with pytest.raises(ValueError):
        run(initial, (probe,), budget=owned_budget)
    assert fake.predictions == fake.inferences == []
