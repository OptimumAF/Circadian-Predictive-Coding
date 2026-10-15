"""Probe-role isolation, explicit damage limits and failure-safe pruning gates."""

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

import src.app.linear_pruning_gate as gate
from src.app.linear_pruning_gate import PruneDamageLimits, gate_linear_prune
from src.app.linear_scaling_measurement import ScalingBatch
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.linear_pc_reference import LinearEquilibrium
from src.core.linear_pc_scaling import (
    LinearScalingConfig,
    LinearScalingState,
    initialize_linear_scaling,
)
from src.core.residual_linear_pc import (
    LinearResidualConfig,
    initialize_residual_linear,
)


def initial_state(family="mlp"):
    if family == "mlp":
        return initialize_linear_scaling(LinearScalingConfig("mean_field", 2, 3, 2), 41)
    return initialize_residual_linear(LinearResidualConfig("mean_field", 2, 3, 2), 41)


def probes():
    return (
        ScalingBatch("train", ("train",), np.array([[0.2, -0.3]]), np.ones((1, 1))),
        ScalingBatch("validation", ("guard",), np.array([[0.7, 0.4]]), np.ones((1, 1))),
    )


def budget(clock=lambda: 0.0, wall=None):
    return ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=0, max_wall_seconds=wall), clock
    )


def run(initial=None, probe_batches=None, **options):
    return gate_linear_prune(
        initial_state() if initial is None else initial,
        options.pop("index", 1),
        probes() if probe_batches is None else probe_batches,
        limits=options.pop("limits", PruneDamageLimits(0.5, 0.0, 0.0)),
        budget=options.pop("budget", budget()),
        **options,
    )


def assert_state_arrays(actual, originals):
    for a, b in zip(actual.raw_weights, originals, strict=True):
        np.testing.assert_array_equal(a, b)


class FakeEvaluation:
    def __init__(self, monkeypatch):
        self.calls = []
        self.fail_at = None
        self.mutate = False
        self.last_record = None
        self.after_call = None
        monkeypatch.setattr(gate, "evaluate_linear_equilibrium", self.evaluate)

    def evaluate(self, weights, features, targets):
        self.calls.append((tuple(w.copy() for w in weights), features.copy(), targets.copy()))
        width = weights[0].shape[1]
        prediction = np.full((len(features), 1), 0.0 if width == 3 else 0.5)
        mse = float(np.mean((prediction - targets) ** 2) / 2)
        result = LinearEquilibrium(
            mse,
            mse,
            1.0,
            prediction,
            tuple(np.zeros((len(features), w.shape[1])) for w in weights[:-1]),
            tuple(np.zeros((len(features), w.shape[1])) for w in weights),
            tuple(np.zeros_like(w) for w in weights),
        )
        self.last_record = result
        if self.mutate:
            for array in (*weights, features, targets):
                array.fill(999)
        if self.after_call is not None:
            self.after_call()
        if len(self.calls) == self.fail_at:
            raise FloatingPointError("deliberate evaluator failure")
        return result


@pytest.mark.parametrize("limit,status", ((0.49, "rejected"), (0.5, "accepted")))
def test_should_complete_every_probe_and_apply_inclusive_independent_limits(
    monkeypatch, limit, status
):
    fake = FakeEvaluation(monkeypatch)
    initial = initial_state()
    originals = tuple(w.copy() for w in initial.raw_weights)
    decision = run(initial, limits=PruneDamageLimits(limit, 0.0, 0.0))
    assert decision.status == status
    assert decision.planned_probe_count == 2
    assert len(decision.diagnostics) == 2 and len(fake.calls) == 4
    assert [d.role for d in decision.diagnostics] == ["train", "validation"]
    for diagnostic in decision.diagnostics:
        assert diagnostic.max_prediction_change == 0.5
        assert diagnostic.source_half_mse == diagnostic.source_energy == 0.5
        assert diagnostic.candidate_half_mse == diagnostic.candidate_energy == 0.125
        assert diagnostic.half_mse_increase == diagnostic.energy_increase == 0
    assert (decision.lineage is not None) == (status == "accepted")
    assert decision.state.config.width == (2 if status == "accepted" else 3)
    assert_state_arrays(initial, originals)
    if status == "rejected":
        assert_state_arrays(decision.state, originals)
    decision.state.raw_weights[0].fill(-999)
    assert_state_arrays(initial, originals)


@pytest.mark.parametrize("limit,status", ((0.24, "rejected"), (0.25, "accepted")))
def test_should_detect_energy_only_damage_with_unchanged_prediction_and_mse(limit, status):
    initial = LinearScalingState(
        LinearScalingConfig("legacy_gaussian", 1, 2, 1),
        (np.zeros((1, 2)), np.array([[0.0], [1.0]])),
    )
    probe = ScalingBatch("validation", ("energy-guard",), np.zeros((1, 1)), np.ones((1, 1)))
    decision = run(initial, (probe,), limits=PruneDamageLimits(0.0, 0.0, limit))
    assert decision.status == status
    assert len(decision.diagnostics) == decision.planned_probe_count == 1
    diagnostic = decision.diagnostics[0]
    assert diagnostic.max_prediction_change == diagnostic.half_mse_increase == 0
    assert diagnostic.source_half_mse == diagnostic.candidate_half_mse == 0.5
    assert diagnostic.source_energy == 0.25 and diagnostic.candidate_energy == 0.5
    assert diagnostic.energy_increase == 0.25


@pytest.mark.parametrize("limit,status", ((15.0, "rejected"), (16.0, "accepted")))
def test_should_report_arbitrary_damage_and_complete_later_probes_even_after_violation(
    limit, status
):
    initial = LinearScalingState(
        LinearScalingConfig("legacy_gaussian", 1, 2, 1),
        (np.array([[1.0, 2.0]]), np.array([[3.0], [4.0]])),
    )
    batches = (
        ScalingBatch("train", ("train",), np.array([[2.0]]), np.zeros((1, 1))),
        ScalingBatch("validation", ("guard",), np.array([[1.0]]), np.zeros((1, 1))),
    )
    decision = run(initial, batches, limits=PruneDamageLimits(limit, 0.0, 0.0))
    assert decision.status == status
    assert [d.max_prediction_change for d in decision.diagnostics] == [16.0, 8.0]
    assert len(decision.diagnostics) == decision.planned_probe_count == 2


@pytest.mark.parametrize("family", ("mlp", "residual"))
def test_should_accept_disconnected_channel_with_real_full_energy_checks(family):
    initial = initial_state(family)
    raw = tuple(w.copy() for w in initial.raw_weights)
    raw[0][:, 1] = 0
    raw[1][1, :] = raw[1][:, 1] = 0
    raw[-1][1, :] = 0
    initial = replace(initial, raw_weights=raw)
    decision = run(initial, limits=PruneDamageLimits(1e-10, 1e-10, 1e-10))
    assert decision.status == "accepted"
    assert decision.state.config.width == 2
    assert decision.lineage is not None
    assert decision.lineage.family == family
    assert len(decision.diagnostics) == 2


def test_should_isolate_each_evaluation_and_private_fallback_from_mutating_evaluator(monkeypatch):
    fake = FakeEvaluation(monkeypatch)
    fake.mutate = True
    initial, batches = initial_state(), probes()
    originals = tuple(w.copy() for w in initial.raw_weights)
    feature_copies = tuple((p.features.copy(), p.targets.copy()) for p in batches)
    decision = run(initial, batches, limits=PruneDamageLimits(0.49, 0.0, 0.0))
    assert decision.status == "rejected"
    assert decision.lineage is None
    assert len(fake.calls) == 4
    for index, (_, x, y) in enumerate(fake.calls):
        np.testing.assert_array_equal(x, feature_copies[index // 2][0])
        np.testing.assert_array_equal(y, feature_copies[index // 2][1])
    assert_state_arrays(initial, originals)
    assert_state_arrays(decision.state, originals)
    for actual, expected in zip(batches, feature_copies, strict=True):
        np.testing.assert_array_equal(actual.features, expected[0])
        np.testing.assert_array_equal(actual.targets, expected[1])


@pytest.mark.parametrize("fail_at,completed", ((2, 0), (4, 1)))
def test_should_retain_only_completed_probe_diagnostics_on_mutating_failure(
    monkeypatch, fail_at, completed
):
    fake = FakeEvaluation(monkeypatch)
    fake.mutate, fake.fail_at = True, fail_at
    initial = initial_state()
    originals = tuple(w.copy() for w in initial.raw_weights)
    decision = run(initial)
    assert decision.status == "failed"
    assert "deliberate" in decision.reason
    assert decision.lineage is None
    assert len(decision.diagnostics) == completed and decision.planned_probe_count == 2
    assert len(fake.calls) == fail_at
    assert_state_arrays(initial, originals)
    assert_state_arrays(decision.state, originals)


def test_should_fail_candidate_creation_without_publishing_a_lineage(monkeypatch):
    fake = FakeEvaluation(monkeypatch)

    def fail(*args):
        raise FloatingPointError("deliberate candidate failure")

    monkeypatch.setattr(gate, "prune_linear_width", fail)
    initial = initial_state()
    originals = tuple(w.copy() for w in initial.raw_weights)
    decision = run(initial)
    assert decision.status == "failed" and decision.lineage is None
    assert decision.diagnostics == () and decision.planned_probe_count == 2
    assert fake.calls == []
    assert_state_arrays(decision.state, originals)


def test_should_copy_evaluation_statistics_before_post_evaluation_callback(monkeypatch):
    fake = FakeEvaluation(monkeypatch)
    owned_budget = budget()
    original_check = owned_budget.before_final

    def callback():
        original_check()
        if fake.last_record is not None:
            fake.last_record.predictions.fill(99)
            object.__setattr__(fake.last_record, "energy", 99.0)
            object.__setattr__(fake.last_record, "mse", 99.0)

    monkeypatch.setattr(owned_budget, "before_final", callback)
    decision = run(budget=owned_budget)
    assert decision.status == "accepted"
    assert all(d.source_energy == d.source_half_mse == 0.5 for d in decision.diagnostics)
    assert all(d.candidate_energy == d.candidate_half_mse == 0.125 for d in decision.diagnostics)
    assert all(d.max_prediction_change == 0.5 for d in decision.diagnostics)


def test_should_copy_source_limits_and_probes_before_budget_callbacks(monkeypatch):
    fake = FakeEvaluation(monkeypatch)
    initial, batches = initial_state(), probes()
    limits = PruneDamageLimits(0.5, 0.0, 0.0)
    old_raw = tuple(w.copy() for w in initial.raw_weights)
    old_probes = tuple((p.features.copy(), p.targets.copy()) for p in batches)
    owned_budget = budget()
    original_check = owned_budget.before_final
    fired = [False]

    def callback():
        original_check()
        if not fired[0]:
            fired[0] = True
            object.__setattr__(initial.config, "gamma0", 2.0)
            object.__setattr__(limits, "max_prediction_change", 0.0)
            for w in initial.raw_weights:
                w.fill(99)
            for p in batches:
                p.features.fill(99)
                p.targets.fill(99)

    monkeypatch.setattr(owned_budget, "before_final", callback)
    decision = run(initial, batches, limits=limits, budget=owned_budget)
    assert fired[0] and decision.status == "accepted"
    for index, (_, x, y) in enumerate(fake.calls):
        np.testing.assert_array_equal(x, old_probes[index // 2][0])
        np.testing.assert_array_equal(y, old_probes[index // 2][1])
    # Source-evaluation first raw coordinates still match copied source geometry.
    assert fake.calls[0][0][0].shape == old_raw[0].shape
    assert decision.lineage is not None
    assert decision.lineage.source_config.gamma0 == 1.0


def test_should_preserve_complete_diagnostics_on_final_budget_failure(monkeypatch):
    fake = FakeEvaluation(monkeypatch)
    owned_budget = budget()
    original_check = owned_budget.before_final
    after_last = [0]

    def callback():
        original_check()
        if len(fake.calls) == 4:
            after_last[0] += 1
            if after_last[0] == 2:
                raise RuntimeError("deliberate final budget failure")

    monkeypatch.setattr(owned_budget, "before_final", callback)
    initial = initial_state()
    originals = tuple(w.copy() for w in initial.raw_weights)
    decision = run(initial, budget=owned_budget)
    assert decision.status == "failed" and decision.lineage is None
    assert len(decision.diagnostics) == decision.planned_probe_count == 2
    assert len(fake.calls) == 4
    assert_state_arrays(decision.state, originals)


def invalid_probes():
    t, v = probes()
    bad: Any = v
    return [
        [],
        (),
        (t,),
        (replace(v, role="final_test"),),
        (replace(v, role="inner_guard"),),
        (replace(v, row_ids=(" ",)),),
        (replace(v, row_ids=("padded ",)),),
        (replace(v, row_ids=("row\x00id",)),),
        (replace(bad, row_ids=["guard"]),),
        (t, replace(v, row_ids=t.row_ids)),
        (t, replace(v, features=t.features)),
        (t, replace(v, targets=t.targets)),
        (t, replace(v, targets=t.features[:, :1])),
        (replace(v, targets=v.features[:, :1]),),
        (replace(v, features=v.features.astype(np.float32)),),
        (replace(v, targets=np.array([[np.inf]])),),
        (replace(v, features=np.ones((1, 3))),),
        (
            replace(
                v,
                features=np.ones((1025, 2)),
                targets=np.ones((1025, 1)),
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


@pytest.mark.parametrize("probe_batches", invalid_probes())
def test_should_reject_invalid_roles_ids_storage_shapes_and_caps_before_callbacks(
    monkeypatch, probe_batches
):
    fake = FakeEvaluation(monkeypatch)
    owned_budget = budget()

    def callback():
        raise AssertionError("preflight invoked budget")

    monkeypatch.setattr(owned_budget, "before_final", callback)
    with pytest.raises(ValueError):
        run(probe_batches=probe_batches, budget=owned_budget)
    assert fake.calls == []


@pytest.mark.parametrize("field", (0, 1, 2))
@pytest.mark.parametrize(
    "value", (True, -1.0, float("inf"), float("nan"), np.float64(0.0), 10**1000)
)
def test_should_require_explicit_native_finite_nonnegative_limits(field, value: Any):
    values = [0.0, 0.0, 0.0]
    values[field] = value
    with pytest.raises(ValueError):
        PruneDamageLimits(*values)


@pytest.mark.parametrize(
    "kind",
    (
        "index_bool",
        "index_bound",
        "limits_family",
        "budget_family",
        "state_family",
        "mutated_state",
    ),
)
def test_should_refuse_bad_controls_or_state_before_callbacks(monkeypatch, kind):
    fake = FakeEvaluation(monkeypatch)
    owned_budget = budget()

    def callback():
        raise AssertionError("invalid public input reached budget")

    monkeypatch.setattr(owned_budget, "before_final", callback)
    initial: Any = initial_state()
    options: dict[str, Any] = dict(budget=owned_budget)
    if kind == "index_bool":
        options["index"] = True
    elif kind == "index_bound":
        options["index"] = 3
    elif kind == "limits_family":
        options["limits"] = {}
    elif kind == "budget_family":
        options["budget"] = None
    elif kind == "state_family":
        initial = object()
    else:
        initial.raw_weights[0][0, 0] = np.nan
    with pytest.raises(ValueError):
        run(initial, **options)
    assert fake.calls == []


def test_should_check_activation_entry_cap_before_budget_callbacks(monkeypatch):
    fake = FakeEvaluation(monkeypatch)
    initial = initialize_linear_scaling(LinearScalingConfig("mean_field", 2, 1000, 1), 41)
    probe = ScalingBatch(
        "validation",
        tuple(f"guard-{i}" for i in range(1000)),
        np.ones((1000, 2)),
        np.ones((1000, 1)),
    )
    owned_budget = budget()

    def callback():
        raise AssertionError("oversized probe invoked budget")

    monkeypatch.setattr(owned_budget, "before_final", callback)
    with pytest.raises(ValueError):
        run(initial, (probe,), budget=owned_budget)
    assert fake.calls == []
