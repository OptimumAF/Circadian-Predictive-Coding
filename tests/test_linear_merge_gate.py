"""Observed linear-merge damage and isolated native-equilibrium probe boundaries.

Core coordinate tolerances remain unchanged. Private scoring helpers are trusted;
fault controls patch their native evaluator rather than fabricate private scores.
"""

from dataclasses import asdict, replace
from math import sqrt
from typing import Any

import numpy as np
import pytest

import src.app.linear_merge_gate as gate
import src.app.linear_pruning_gate as native_scores
from src.app.linear_merge_gate import MergeDamageLimits, MergeProbeDiagnostic, gate_linear_merge
from src.app.linear_pruning_gate import PruneDamageLimits, PruneProbeDiagnostic
from src.app.linear_scaling_measurement import ScalingBatch
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.linear_pc_reference import LinearEquilibrium, evaluate_linear_equilibrium
from src.core.linear_pc_scaling import (
    LinearScalingConfig,
    LinearScalingState,
    effective_linear_weights,
    initialize_linear_scaling,
    step_linear_pc,
)
from src.core.linear_width_structure import (
    LinearWidthLineage,
    merge_linear_width,
    split_linear_width,
)
from src.core.residual_linear_pc import (
    LinearResidualConfig,
    LinearResidualState,
    effective_residual_weights,
    initialize_residual_linear,
    step_residual_linear_pc,
)


def initial(family="mlp", scheme="mean_field", hidden=2, width=3):
    gamma = 2.0 if scheme == "mean_field" else 1.0
    if family == "mlp":
        return initialize_linear_scaling(LinearScalingConfig(scheme, 2, width, hidden, gamma), 41)
    return initialize_residual_linear(LinearResidualConfig(scheme, 2, width, hidden, gamma), 41)


def effective(state):
    return (
        effective_linear_weights(state)
        if type(state) is LinearScalingState
        else effective_residual_weights(state)
    )


def probes(target=1.0):
    return (
        ScalingBatch("train", ("train",), np.array([[0.2, -0.3]]), np.array([[target]])),
        ScalingBatch("validation", ("guard",), np.array([[0.7, 0.4]]), np.array([[target]])),
    )


def budget():
    return ToyBudgetSession(ToyExecutionBudget(max_training_updates=0), lambda: 0.0)


def run(state=None, lineage=None, data=None, **options):
    if state is None:
        split = split_linear_width(initial(), 1)
        state, lineage = split.state, split.lineage
    return gate_linear_merge(
        state,
        lineage,
        probes() if data is None else data,
        limits=options.pop("limits", MergeDamageLimits(1.0, 1.0, 1.0)),
        budget=options.pop("budget", budget()),
        **options,
    )


def dense_merge_reference(state, lineage):
    n = lineage.source_config.width
    s = np.zeros((n, n + 1))
    s[:, :n] = np.eye(n)
    s[lineage.split_index, lineage.split_index] = s[lineage.split_index, -1] = 1 / sqrt(2)
    a = effective(state)
    middle = tuple(
        s @ w @ s.T
        if type(state) is LinearScalingState
        else np.eye(n) + s @ (w - np.eye(n + 1)) @ s.T
        for w in a[1:-1]
    )
    return (a[0] @ s.T, *middle, s @ a[-1])


def analytical(weights, x, y):
    prediction = x.copy()
    for w in weights:
        prediction = prediction @ w
    mse = float(np.mean((prediction - y) ** 2) / 2)
    suffix = weights[-1]
    scale = 1 + float(np.sum(suffix * suffix))
    for w in reversed(weights[1:-1]):
        suffix = w @ suffix
        scale += float(np.sum(suffix * suffix))
    return prediction, mse, mse / scale


def test_should_use_exact_existing_public_limit_and_diagnostic_class_aliases():
    assert MergeDamageLimits is PruneDamageLimits
    assert MergeProbeDiagnostic is PruneProbeDiagnostic


@pytest.mark.parametrize("family", ("mlp", "residual"))
@pytest.mark.parametrize("hidden", (1, 2, 3))
@pytest.mark.parametrize("scheme", ("legacy_gaussian", "fan_in", "mean_field"))
def test_should_match_independent_dense_reencoding_and_native_full_energy_for_every_probe(
    family, hidden, scheme
):
    original = initial(family, scheme, hidden)
    split = split_linear_width(original, 1)
    before = asdict(split.state)
    expected = dense_merge_reference(split.state, split.lineage)
    data = probes()
    decision = run(split.state, split.lineage, data, limits=MergeDamageLimits(1e-10, 1e-10, 1e-10))
    assert decision.status == "accepted" and decision.lineage is not None
    assert decision.state.config == original.config and type(decision.state) is type(original)
    for index, (actual, wanted, raw, scale) in enumerate(
        zip(
            effective(decision.state),
            expected,
            decision.state.raw_weights,
            original.config.forward_scales,
            strict=True,
        )
    ):
        np.testing.assert_allclose(actual, wanted, rtol=3e-13, atol=3e-14)
        branch = wanted - (
            np.eye(original.config.width)
            if family == "residual" and 0 < index < len(expected) - 1
            else 0
        )
        np.testing.assert_allclose(raw, branch / scale, rtol=3e-13, atol=3e-14)
    assert len(decision.diagnostics) == decision.planned_probe_count == 2
    for diagnostic, probe in zip(decision.diagnostics, data, strict=True):
        a = evaluate_linear_equilibrium(effective(split.state), probe.features, probe.targets)
        b = evaluate_linear_equilibrium(expected, probe.features, probe.targets)
        for observed, record, weights in (
            (diagnostic.source_energy, a, effective(split.state)),
            (diagnostic.candidate_energy, b, expected),
        ):
            pred, mse, energy = analytical(weights, probe.features, probe.targets)
            np.testing.assert_allclose(record.predictions, pred, rtol=2e-13, atol=2e-14)
            assert record.mse == pytest.approx(mse, abs=2e-14)
            assert observed == pytest.approx(energy, abs=2e-14)
        assert diagnostic.source_half_mse == pytest.approx(a.mse, abs=2e-14)
        assert diagnostic.candidate_half_mse == pytest.approx(b.mse, abs=2e-14)
        assert diagnostic.max_prediction_change == pytest.approx(
            float(np.max(np.abs(a.predictions - b.predictions))), abs=2e-14
        )
    np.testing.assert_equal(asdict(split.state), before)


@pytest.mark.parametrize("family", ("mlp", "residual"))
def test_should_reject_original_compatible_amplification_even_when_both_native_losses_improve(
    family,
):
    config: Any
    wide: Any
    if family == "mlp":
        config = LinearScalingConfig("legacy_gaussian", 1, 1, 1)
        wide = LinearScalingState(
            replace(config, width=2),
            (np.array([[1e-13, -1e-13]]), np.array([[1e20 + 1e9], [1e20 - 1e9]])),
        )
    else:
        config = LinearResidualConfig("legacy_gaussian", 1, 1, 2)
        wide = LinearResidualState(
            replace(config, width=2),
            (np.array([[1e-13, -1e-13]]), np.zeros((2, 2)), np.array([[1e20 + 1e9], [1e20 - 1e9]])),
        )
    lineage = LinearWidthLineage(1, family, config, 0)
    merged = merge_linear_width(wide, lineage)
    assert merged.config.width == 1
    data = tuple(
        ScalingBatch(role, (name,), np.ones((1, 1)), np.zeros((1, 1)))
        for role, name in (("train", "amp-train"), ("validation", "amp-guard"))
    )
    decision = run(wide, lineage, data, limits=MergeDamageLimits(1e-6, 0.0, 0.0))
    assert decision.status == "rejected" and decision.lineage is None
    assert len(decision.diagnostics) == 2
    for d in decision.diagnostics:
        assert d.max_prediction_change > 1e-6
        assert d.candidate_half_mse < d.source_half_mse
        assert d.candidate_energy < d.source_energy
        assert d.half_mse_increase == d.energy_increase == 0
    np.testing.assert_equal(asdict(decision.state), asdict(wide))


class FakeEvaluation:
    def __init__(self, monkeypatch, mode="prediction"):
        self.calls = []
        self.last = None
        self.mutate = False
        self.fail_at = None
        self.override = None
        self.mode = mode
        monkeypatch.setattr(native_scores, "evaluate_linear_equilibrium", self.evaluate)

    def evaluate(self, weights, x, y):
        self.calls.append((tuple(w.copy() for w in weights), x.copy(), y.copy()))
        candidate = weights[0].shape[1] == 3
        value = (
            (0.5 if candidate else 0.0)
            if self.mode == "prediction"
            else ((1.0 if candidate else 0.0) if self.mode == "mse" else 0.0)
        )
        prediction = np.full((len(x), 1), value)
        mse = float(np.mean((prediction - y) ** 2) / 2)
        energy = (0.5 if candidate else 0.25) if self.mode == "energy" else mse
        record = LinearEquilibrium(
            energy,
            mse,
            mse / energy if energy else 1.0,
            prediction,
            tuple(np.zeros((len(x), w.shape[1])) for w in weights[:-1]),
            tuple(np.zeros((len(x), w.shape[1])) for w in weights),
            tuple(np.zeros_like(w) for w in weights),
        )
        if self.override:
            record = self.override(record)
        self.last = record
        if self.mutate:
            for a in (*weights, x, y):
                a[:] = 999
        if len(self.calls) == self.fail_at:
            raise FloatingPointError("evaluator fault")
        return record


@pytest.mark.parametrize(
    "mode,threshold,status",
    (
        ("prediction", 0.49, "rejected"),
        ("prediction", 0.5, "accepted"),
        ("mse", 0.49, "rejected"),
        ("mse", 0.5, "accepted"),
        ("energy", 0.24, "rejected"),
        ("energy", 0.25, "accepted"),
    ),
)
def test_should_complete_all_probes_and_apply_each_observed_limit_inclusively(
    monkeypatch, mode, threshold, status
):
    fake = FakeEvaluation(monkeypatch, mode)
    limits = {
        "prediction": MergeDamageLimits(threshold, 1, 1),
        "mse": MergeDamageLimits(1, threshold, 1),
        "energy": MergeDamageLimits(0, 0, threshold),
    }[mode]
    result = run(data=probes(0.0 if mode == "mse" else 1.0), limits=limits)
    assert result.status == status and len(fake.calls) == 4
    assert len(result.diagnostics) == result.planned_probe_count == 2
    if mode == "energy":
        assert all(
            d.max_prediction_change == d.half_mse_increase == 0 and d.energy_increase == 0.25
            for d in result.diagnostics
        )
    assert (result.lineage is not None) == (status == "accepted")


def test_should_own_every_native_evaluation_input_and_copy_outputs_before_budget_callbacks(
    monkeypatch,
):
    fake = FakeEvaluation(monkeypatch)
    fake.mutate = True
    split = split_linear_width(initial(), 1)
    before = asdict(split.state)
    data = probes()
    copies = tuple((p.features.copy(), p.targets.copy()) for p in data)
    session = budget()

    def mutate_record():
        if fake.last is not None:
            fake.last.predictions[:] = 998
            object.__setattr__(fake.last, "energy", 997.0)
            object.__setattr__(fake.last, "mse", 996.0)

    monkeypatch.setattr(session, "before_final", mutate_record)
    result = run(
        split.state, split.lineage, data, budget=session, limits=MergeDamageLimits(0.49, 0, 0)
    )
    assert result.status == "rejected" and result.lineage is None
    assert len(fake.calls) == 4
    for index, call in enumerate(fake.calls):
        np.testing.assert_array_equal(call[1], copies[index // 2][0])
        np.testing.assert_array_equal(call[2], copies[index // 2][1])
    for a, b in ((fake.calls[0][0], fake.calls[2][0]), (fake.calls[1][0], fake.calls[3][0])):
        for u, v in zip(a, b, strict=True):
            np.testing.assert_array_equal(u, v)
    assert all(
        d.max_prediction_change == 0.5 and d.source_half_mse == 0.5 for d in result.diagnostics
    )
    np.testing.assert_equal(asdict(result.state), before)
    np.testing.assert_equal(asdict(split.state), before)


@pytest.mark.parametrize("fail_at", (1, 2, 3, 4))
def test_should_keep_only_complete_probe_prefix_and_private_fallback_after_mutating_native_fault(
    monkeypatch, fail_at
):
    fake = FakeEvaluation(monkeypatch)
    fake.mutate = True
    fake.fail_at = fail_at
    split = split_linear_width(initial(), 1)
    before = asdict(split.state)
    result = run(split.state, split.lineage)
    assert result.status == "failed" and "evaluator fault" in result.reason
    assert result.lineage is None and len(result.diagnostics) == (fail_at - 1) // 2
    np.testing.assert_equal(asdict(result.state), before)
    np.testing.assert_equal(asdict(split.state), before)


def test_should_copy_source_lineage_data_and_limits_before_budget_mutates_originals(monkeypatch):
    fake = FakeEvaluation(monkeypatch)
    split = split_linear_width(initial(), 1)
    before = asdict(split.state)
    data = probes()
    limits = MergeDamageLimits(0.49, 0, 0)
    session = budget()

    def mutate():
        split.state.raw_weights[0][:] = 999
        object.__setattr__(split.lineage.source_config, "gamma0", 3.0)
        for p in data:
            p.features[:] = 999
            p.targets[:] = 999
        object.__setattr__(limits, "max_prediction_change", 1.0)

    monkeypatch.setattr(session, "before_final", mutate)
    result = run(split.state, split.lineage, data, limits=limits, budget=session)
    assert result.status == "rejected" and result.lineage is None
    np.testing.assert_equal(asdict(result.state), before)
    for index, call in enumerate(fake.calls):
        np.testing.assert_array_equal(call[1], probes()[index // 2].features)
        np.testing.assert_array_equal(call[2], probes()[index // 2].targets)


def test_should_pass_private_kernel_arguments_and_detach_accepted_result_and_lineage(monkeypatch):
    FakeEvaluation(monkeypatch)
    split = split_linear_width(initial(), 1)
    before, metadata = asdict(split.state), asdict(split.lineage)
    captured = []

    def kernel(state, lineage):
        result = merge_linear_width(state, lineage)
        captured.append(result)
        state.raw_weights[0][:] = 999
        object.__setattr__(lineage.source_config, "gamma0", 3.0)
        return result

    monkeypatch.setattr(gate, "merge_linear_width", kernel)
    result = run(split.state, split.lineage)
    assert result.status == "accepted" and result.lineage is not None
    captured[0].raw_weights[0][:] = 998
    assert not np.any(result.state.raw_weights[0] == 998)
    object.__setattr__(result.lineage.source_config, "gamma0", 4.0)
    np.testing.assert_equal(asdict(split.state), before)
    np.testing.assert_equal(asdict(split.lineage), metadata)


@pytest.mark.parametrize("bad", ("family", "config", "dtype", "nan", "exception"))
def test_should_keep_source_and_no_scores_when_merge_candidate_is_invalid(monkeypatch, bad):
    fake = FakeEvaluation(monkeypatch)
    split = split_linear_width(initial(), 1)
    before = asdict(split.state)

    def malformed(state, lineage):
        result = merge_linear_width(state, lineage)
        if bad == "family":
            return initial("residual")
        if bad == "config":
            object.__setattr__(result.config, "gamma0", 3.0)
        elif bad == "dtype":
            object.__setattr__(
                result, "raw_weights", tuple(w.astype(np.float32) for w in result.raw_weights)
            )
        elif bad == "nan":
            result.raw_weights[0][0, 0] = np.nan
        else:
            state.raw_weights[0][:] = 999
            raise RuntimeError("kernel fault")
        return result

    monkeypatch.setattr(gate, "merge_linear_width", malformed)
    result = run(split.state, split.lineage)
    assert result.status == "failed" and result.lineage is None and result.diagnostics == ()
    assert fake.calls == []
    np.testing.assert_equal(asdict(result.state), before)


@pytest.mark.parametrize("bad", ("family", "energy", "mse", "dtype", "shape", "prediction"))
def test_should_refuse_nonfinite_or_malformed_native_equilibrium_outputs(monkeypatch, bad):
    fake = FakeEvaluation(monkeypatch)

    def override(record):
        if bad == "family":
            return asdict(record)
        if bad == "energy":
            return replace(record, energy=float("nan"))
        if bad == "mse":
            return replace(record, mse=-1.0)
        if bad == "dtype":
            return replace(record, predictions=record.predictions.astype(np.float32))
        if bad == "shape":
            return replace(record, predictions=record.predictions.ravel())
        return replace(record, predictions=np.full_like(record.predictions, np.inf))

    fake.override = override
    result = run()
    assert result.status == "failed" and result.lineage is None and result.diagnostics == ()


def test_should_keep_complete_diagnostics_but_no_authority_on_final_budget_fault(monkeypatch):
    fake = FakeEvaluation(monkeypatch)
    session = budget()

    def fail_late():
        if len(fake.calls) == 4:
            raise RuntimeError("late budget fault")

    monkeypatch.setattr(session, "before_final", fail_late)
    result = run(budget=session)
    assert result.status == "failed" and result.lineage is None
    assert len(result.diagnostics) == 2 and result.planned_probe_count == 2


def test_should_reuse_unchanged_source_after_failed_assessment(monkeypatch):
    fake = FakeEvaluation(monkeypatch)
    fake.fail_at = 2
    fake.mutate = True
    split = split_linear_width(initial(), 1)
    before = asdict(split.state)
    failed = run(split.state, split.lineage)
    assert failed.status == "failed"
    fake.fail_at = None
    fake.calls.clear()
    accepted = run(split.state, split.lineage)
    assert accepted.status == "accepted" and accepted.lineage is not None
    np.testing.assert_equal(asdict(split.state), before)


def invalid_probes():
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
        (replace(a, targets=np.ones(1)), b),
        (replace(a, features=np.full((1, 2), np.nan)), b),
        (replace(a, features=np.ones((1, 3))), b),
        (a, replace(b, features=a.features)),
        (replace(a, targets=a.features[:, :1]), b),
        (
            replace(
                a,
                features=np.ones((1025, 2)),
                targets=np.ones((1025, 1)),
                row_ids=tuple(f"row-{i}" for i in range(1025)),
            ),
            b,
        ),
    ]


@pytest.mark.parametrize("data", invalid_probes())
def test_should_preflight_declared_roles_ids_storage_and_shapes_before_callbacks(monkeypatch, data):
    fake = FakeEvaluation(monkeypatch)
    session = budget()
    calls = []
    monkeypatch.setattr(session, "before_final", lambda: calls.append("budget"))
    with pytest.raises(ValueError):
        run(data=data, budget=session)
    assert fake.calls == [] and calls == []


def test_should_refuse_final_role_before_accessing_or_copying_sentinel_payload():
    class Sentinel:
        def __deepcopy__(self, memo):
            raise AssertionError("final payload copied")

    a, b = probes()
    with pytest.raises(ValueError):
        run(data=(replace(a, role="final_test", features=Sentinel(), targets=Sentinel()), b))


@pytest.mark.parametrize(
    "bad",
    (
        "state_family",
        "lineage_family",
        "format",
        "source_config",
        "width",
        "index",
        "limits",
        "budget",
    ),
)
def test_should_validate_exact_family_format_full_configuration_and_controls_before_budget(
    monkeypatch, bad
):
    fake = FakeEvaluation(monkeypatch)
    split = split_linear_width(initial(), 1)
    state: Any = split.state
    lineage: Any = split.lineage
    options: dict[str, Any] = {}
    session = budget()
    calls = []
    monkeypatch.setattr(session, "before_final", lambda: calls.append("budget"))
    if bad == "state_family":
        state = asdict(state)
    elif bad == "lineage_family":
        lineage = asdict(lineage)
    elif bad == "format":
        object.__setattr__(lineage, "format_version", True)
    elif bad == "source_config":
        object.__setattr__(lineage.source_config, "gamma0", 3.0)
    elif bad == "width":
        object.__setattr__(state.config, "width", 3)
    elif bad == "index":
        object.__setattr__(lineage, "split_index", True)
    elif bad == "limits":
        options["limits"] = {}
    else:
        session = {}
    with pytest.raises(ValueError):
        run(state, lineage, budget=session, **options)
    assert calls == [] and fake.calls == []


def test_should_enforce_probe_activation_cap_before_budget_or_native_equilibrium(monkeypatch):
    fake = FakeEvaluation(monkeypatch)
    split = split_linear_width(initial(hidden=10, width=100), 1)
    data = (
        ScalingBatch(
            "validation",
            tuple(f"many-{i}" for i in range(1024)),
            np.ones((1024, 2)),
            np.ones((1024, 1)),
        ),
    )
    session = budget()
    calls = []
    monkeypatch.setattr(session, "before_final", lambda: calls.append("budget"))
    with pytest.raises(ValueError):
        run(split.state, split.lineage, data, budget=session)
    assert calls == [] and fake.calls == []


@pytest.mark.parametrize("family", ("mlp", "residual"))
def test_should_continue_only_train_data_on_detached_accepted_state_without_changing_source(family):
    split = split_linear_width(initial(family), 1)
    before = asdict(split.state)
    decision = run(split.state, split.lineage, limits=MergeDamageLimits(1e-10, 1e-10, 1e-10))
    assert decision.status == "accepted" and decision.lineage is not None
    train = probes()[0]
    copied = type(decision.state)(decision.state.config, decision.state.raw_weights)
    step = step_linear_pc if family == "mlp" else step_residual_linear_pc
    a = step(decision.state, train.features, train.targets, 0.01)
    b = step(copied, train.features, train.targets, 0.01)
    np.testing.assert_equal(asdict(a.state), asdict(b.state))
    np.testing.assert_equal(asdict(split.state), before)
