"""Bounded nonlinear runner isolation, with one eight-update native control.

Fault and preflight cases replace diagnostics and use a raw-state spy; they make
no parameter-kernel calls. Final labels are consumed only by delayed app scoring.
"""

from dataclasses import asdict, replace
from typing import Any

import numpy as np
import pytest

import src.app.nonlinear_scaling_measurement as measurement
from src.adapters.coupled_scaling_learner import CoupledScalingLearner
from src.adapters.scaled_matched_backprop_learner import ScaledMatchedBackpropLearner
from src.app.coupled_attribution import AttributionBatch
from src.app.nonlinear_scaling_measurement import run_nonlinear_scaling_case
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.coupled_circadian_predictive_coding import CoupledCircadianConfig
from src.core.coupled_energy import evaluate_coupled, feedforward_latents
from src.core.coupled_pc_scaling import (
    CoupledScalingConfig,
    copy_coupled_scaling_state,
    effective_coupled_parameters,
    initialize_coupled_scaling,
)
from src.core.learner_ports import TrainingDiagnostic
from src.core.matched_backprop import evaluate_matched_backprop


def batches():
    training = (
        AttributionBatch("train", "A", ("train-0",), np.array([[0.2, -0.3]]), np.array([[0.2]])),
        AttributionBatch("train", "A", ("train-1",), np.array([[0.7, 0.4]]), np.array([[0.8]])),
    )
    heldout = (
        AttributionBatch(
            "final_test", "A", ("test-0",), np.array([[1.1, -0.8]]), np.array([[0.6]])
        ),
    )
    return training, heldout


def initial(method="bp", **changes):
    policy = (
        None
        if method in ("bp", "pc")
        else (
            CoupledCircadianConfig.matched_pc_control()
            if method == "neutral_cpc"
            else CoupledCircadianConfig()
        )
    )
    values: dict[str, Any] = dict(
        scheme="mean_field",
        input_dim=2,
        width=3,
        hidden_layers=2,
        gamma0=2.0,
        precisions=(0.7, 1.3),
        circadian=policy,
    )
    values.update(changes)
    return initialize_coupled_scaling(CoupledScalingConfig(**values), 41)


def arrays(p):
    return (*p.hidden_weights, *p.hidden_biases, p.output_weight, p.output_bias)


def budget(updates=2, clock=lambda: 0.0, wall=None):
    return ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=updates, max_wall_seconds=wall), clock
    )


class SpyLearner:
    def __init__(self, state=None, trace=None, control=None):
        self.state = initial() if state is None else copy_coupled_scaling_state(state)
        self.trace = [] if trace is None else trace
        self.control = {} if control is None else control

    def fork(self):
        self.trace.append(("fork",))
        return SpyLearner(self.state, self.trace, self.control)

    def snapshot_state(self):
        self.trace.append(("snapshot",))
        if self.control.get("snapshot_callback"):
            callback = self.control.pop("snapshot_callback")
            callback()
        return copy_coupled_scaling_state(self.state)

    def restore_state(self, state):
        self.state = copy_coupled_scaling_state(state)

    def train_batch(self, x, y):
        self.trace.append(("train", x.copy(), y.copy()))
        if self.control.get("fail_train") == len([e for e in self.trace if e[0] == "train"]):
            raise RuntimeError("training fault")
        self.state.raw_parameters.output_bias[:] += 1
        if self.control.get("after_train"):
            self.control["after_train"]()
        x[:] = 991
        y[:] = 992
        return TrainingDiagnostic("measurement_spy_v1", 0.7)

    def predict(self, x):
        self.trace.append(("predict", float(self.state.raw_parameters.output_bias[0, 0]), x.copy()))
        if self.control.get("on_predict"):
            self.control["on_predict"]()
        prediction = np.full((len(x), 1), 0.2 + 0.1 * self.state.raw_parameters.output_bias[0, 0])
        if "prediction" in self.control:
            prediction = self.control["prediction"](x)
        x[:] = 993
        return prediction


def run(source=None, training=None, heldout=None, **options):
    default = batches()
    return run_nonlinear_scaling_case(
        SpyLearner() if source is None else source,
        default[0] if training is None else training,
        default[1] if heldout is None else heldout,
        method=options.pop("method", "bp"),
        checkpoint_updates=options.pop("checkpoint_updates", (0, 1, 2)),
        label_delay=options.pop("label_delay", 3),
        budget=options.pop("budget", budget()),
        clock=options.pop("clock", lambda: 0.0),
        **options,
    )


def fake_diagnostics(monkeypatch, source, *, fail_at=None, now=None):
    def diagnose(state, batch, updates, method, *, initial_features=None):
        if updates == 0:
            assert batch is None
            np.testing.assert_array_equal(initial_features, batches()[0][0].features)
            initial_features[:] = 984
        else:
            assert initial_features is None
        source.trace.append(
            (
                "diagnostic",
                updates,
                None if batch is None else batch.features.copy(),
                None if batch is None else batch.targets.copy(),
            )
        )
        if now is not None:
            now[0] += 0.5
        if updates == fail_at:
            raise FloatingPointError("diagnostic fault")
        state.raw_parameters.output_bias[:] = 987
        if batch is not None:
            assert batch.role == "train"
            batch.features[:] = 985
            batch.targets[:] = 986
        return dict(
            updates=updates,
            energy=None if batch is None else 0.7,
            training_context=None if batch is None else batch.context,
            row_ids=None if batch is None else batch.row_ids,
        )

    monkeypatch.setattr(measurement, "_diagnostics", diagnose)


def test_should_observe_only_arrived_training_labels_and_score_detached_states_after_all_training(
    monkeypatch,
):
    source = SpyLearner()
    training, heldout = batches()
    original = asdict(source.state)
    data_before = [a.copy() for b in (*training, *heldout) for a in (b.features, b.targets)]
    now = [0.0]
    fake_diagnostics(monkeypatch, source, now=now)
    result = run(source, training, heldout, clock=lambda: now[0])
    assert result["status"] == "complete" and result["committed_updates"] == 2
    assert [r["updates"] for r in result["diagnostics"]] == [0, 1, 2]
    diagnostics = [e for e in source.trace if e[0] == "diagnostic"]
    assert diagnostics[0][2:] == (None, None)
    for event, batch in zip(diagnostics[1:], training, strict=True):
        np.testing.assert_array_equal(event[2], batch.features)
        np.testing.assert_array_equal(event[3], batch.targets)
    train = [e for e in source.trace if e[0] == "train"]
    predictions = [e for e in source.trace if e[0] == "predict"]
    assert len(train) == 2 and [e[1] for e in predictions] == [0, 1, 2]
    assert max(i for i, e in enumerate(source.trace) if e[0] == "train") < min(
        i for i, e in enumerate(source.trace) if e[0] == "predict"
    )
    assert [e["arrived_at"] for e in result["stream_result"]["applied_updates"]] == [3, 4]
    for event, batch in zip(train, training, strict=True):
        np.testing.assert_array_equal(event[1], batch.features)
        np.testing.assert_array_equal(event[2], batch.targets)
    for actual, expected in zip(
        (a for b in (*training, *heldout) for a in (b.features, b.targets)),
        data_before,
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_equal(asdict(source.state), original)
    assert result["diagnostic_seconds"] == pytest.approx(1.5)
    assert result["elapsed_seconds"] >= result["diagnostic_seconds"]
    result["initial_state"].raw_parameters.output_bias[:] = 998
    result["final_state"].raw_parameters.output_bias[:] = 999
    np.testing.assert_equal(asdict(source.state), original)


def test_should_copy_all_batches_before_first_external_snapshot_callback(monkeypatch):
    training, heldout = batches()
    source = SpyLearner()

    def mutate():
        for batch in (*training, *heldout):
            batch.features[:] = 777
            batch.targets[:] = 0

    source.control["snapshot_callback"] = mutate
    fake_diagnostics(monkeypatch, source)
    result = run(source, training, heldout)
    assert result["status"] == "complete"
    for event, expected in zip(
        (e for e in source.trace if e[0] == "train"), batches()[0], strict=True
    ):
        np.testing.assert_array_equal(event[1], expected.features)
        np.testing.assert_array_equal(event[2], expected.targets)
    for event in (e for e in source.trace if e[0] == "predict"):
        np.testing.assert_array_equal(event[2], batches()[1][0].features)


@pytest.mark.parametrize("fail_at", (0, 1, 2))
def test_should_keep_known_commits_and_completed_diagnostics_but_no_scores_after_diagnostic_failure(
    monkeypatch, fail_at
):
    source = SpyLearner()
    fake_diagnostics(monkeypatch, source, fail_at=fail_at)
    result = run(source)
    assert result["status"] == "failed" and result["committed_updates"] == fail_at
    assert result["stream_result"] is None
    assert [r["updates"] for r in result["diagnostics"] if r["status"] == "complete"] == list(
        range(fail_at)
    )
    assert result["diagnostics"][-1]["updates"] == fail_at
    assert result["diagnostics"][-1]["status"] == "failed"
    assert "diagnostic fault" in result["diagnostics"][-1]["reason"]
    assert not any(e[0] == "predict" for e in source.trace)


@pytest.mark.parametrize("fail_at", (1, 2))
def test_should_report_committed_work_before_a_failed_training_call(monkeypatch, fail_at):
    source = SpyLearner(control={"fail_train": fail_at})
    fake_diagnostics(monkeypatch, source)
    result = run(source)
    assert result["status"] == "failed" and result["committed_updates"] == fail_at - 1
    assert result["stream_result"] is None
    assert not any(e[0] == "predict" for e in source.trace)


def test_should_clear_all_scores_if_last_score_exhausts_budget(monkeypatch):
    now = [0.0]
    source = SpyLearner()
    count = [0]

    def expire():
        count[0] += 1
        if count[0] == 3:
            now[0] = 10.0

    source.control["on_predict"] = expire
    fake_diagnostics(monkeypatch, source)
    result = run(source, budget=budget(clock=lambda: now[0], wall=1.0))
    assert result["status"] == "failed" and result["committed_updates"] == 2
    assert count[0] == 3 and result["stream_result"] is None


def test_should_retain_last_commit_when_clock_fails_after_native_return(monkeypatch):
    source = SpyLearner()
    flag = [False]
    source.control["after_train"] = lambda: flag.__setitem__(0, True)

    def clock():
        if flag[0]:
            raise RuntimeError("clock fault")
        return 0.0

    fake_diagnostics(monkeypatch, source)
    result = run(source, clock=clock)
    assert result["status"] == "failed" and result["committed_updates"] == 1
    assert result["stream_result"] is None
    assert not any(e[0] == "predict" for e in source.trace)


@pytest.mark.parametrize("values", ((float("nan"),), (0.0, 0.2, 0.1)))
def test_should_report_invalid_or_reversing_clocks_without_scores(monkeypatch, values):
    source = SpyLearner()
    fake_diagnostics(monkeypatch, source)
    readings = iter(values)
    last = [values[-1]]

    def clock():
        last[0] = next(readings, last[0])
        return last[0]

    result = run(source, clock=clock)
    assert result["status"] == "failed" and result["stream_result"] is None
    assert not any(e[0] == "predict" for e in source.trace)
    if np.isnan(values[0]):
        assert result["elapsed_seconds"] is result["diagnostic_seconds"] is None
        assert result["timing_status"] == "unavailable"
        assert result["committed_updates"] == 0 and result["diagnostics"] == []


@pytest.mark.parametrize(
    "prediction",
    (
        lambda x: np.ones((len(x), 1), dtype=np.float32),
        lambda x: np.ones(len(x)),
        lambda x: np.full((len(x), 1), np.nan),
        lambda x: np.full((len(x), 1), 1.1),
    ),
)
def test_should_refuse_invalid_target_free_predictions_without_partial_final_scores(
    monkeypatch, prediction
):
    source = SpyLearner(control={"prediction": prediction})
    fake_diagnostics(monkeypatch, source)
    result = run(source)
    assert result["status"] == "failed" and result["committed_updates"] == 2
    assert result["stream_result"] is None


def invalid_batches():
    t, h = batches()
    a, b = t
    c = h[0]
    return [
        ((), h),
        (list(t), h),
        (t * 5, h),
        (t, ()),
        (t, h * 5),
        ((replace(a, role="final_test"), b), h),
        (t, (replace(c, role="validation"),)),
        ((replace(a, row_ids=(" padded ",)), b), h),
        ((replace(a, row_ids=("line\nbreak",)), b), h),
        ((replace(a, row_ids=b.row_ids), b), h),
        (t, (replace(c, row_ids=a.row_ids),)),
        ((replace(a, features=a.features.astype(np.float32)), b), h),
        ((replace(a, targets=np.array([[2.0]])), b), h),
        ((replace(a, targets=np.ones(1)), b), h),
        ((replace(a, features=np.full((1, 2), np.nan)), b), h),
        ((replace(a, features=np.ones((1, 3))), b), h),
        (
            (
                replace(
                    a,
                    features=np.ones((257, 2)),
                    targets=np.ones((257, 1)),
                    row_ids=tuple(f"large-{i}" for i in range(257)),
                ),
                b,
            ),
            h,
        ),
        ((a, replace(b, features=a.features)), h),
        (t, (replace(c, targets=a.targets),)),
        ((replace(a, targets=a.features[:, :1]), b), h),
    ]


@pytest.mark.parametrize("training,heldout", invalid_batches())
def test_should_refuse_invalid_data_and_all_storage_aliases_before_callbacks(
    monkeypatch, training, heldout
):
    source = SpyLearner()
    session = budget()
    calls = []
    monkeypatch.setattr(session, "before_final", lambda: calls.append("budget"))
    with pytest.raises(ValueError):
        run(source, training, heldout, budget=session)
    assert source.trace == [] and calls == []


def test_should_refuse_final_role_before_copying_sentinel_or_budget_callbacks(monkeypatch):
    class Sentinel:
        def __deepcopy__(self, memo):
            raise AssertionError("reserved data copied")

    training, heldout = batches()
    source = SpyLearner()
    session = budget()
    calls = []
    monkeypatch.setattr(session, "before_final", lambda: calls.append("budget"))
    bad = replace(training[0], role="final_test", features=Sentinel(), targets=Sentinel())
    with pytest.raises(ValueError):
        run(source, (bad, training[1]), heldout, budget=session)
    assert source.trace == [] and calls == []


@pytest.mark.parametrize(
    "method,source_method",
    (
        ("unknown", "bp"),
        ("pc", "active_cpc"),
        ("bp", "neutral_cpc"),
        ("neutral_cpc", "active_cpc"),
        ("active_cpc", "neutral_cpc"),
    ),
)
def test_should_refuse_method_and_native_policy_family_mismatch_before_work(
    monkeypatch, method, source_method
):
    source = SpyLearner(initial(source_method))
    fake_diagnostics(monkeypatch, source)
    with pytest.raises(ValueError):
        run(source, method=method)
    assert not any(e[0] in ("train", "diagnostic", "predict") for e in source.trace)


@pytest.mark.parametrize(
    "changes", ({"input_dim": 17}, {"width": 33}, {"hidden_layers": 5, "precisions": (1.0,) * 5})
)
def test_should_enforce_measurement_geometry_before_native_work(monkeypatch, changes):
    source = SpyLearner(initial(**changes))
    fake_diagnostics(monkeypatch, source)
    with pytest.raises(ValueError):
        run(source)
    assert not any(e[0] in ("train", "diagnostic", "predict") for e in source.trace)


@pytest.mark.parametrize(
    "options",
    (
        {"label_delay": True},
        {"label_delay": -1},
        {"checkpoint_updates": (0, 1)},
        {"checkpoint_updates": [0, 2]},
        {"budget": {}},
        {"clock": 3},
    ),
)
def test_should_refuse_invalid_controls_before_callbacks(options):
    source = SpyLearner()
    with pytest.raises(ValueError):
        run(source, **options)
    assert source.trace == []


def test_should_measure_four_real_methods_from_identical_owned_initial_parameters():
    results = {}
    training, heldout = batches()
    for method in ("bp", "pc", "neutral_cpc", "active_cpc"):
        state = initial(method)
        source = (
            ScaledMatchedBackpropLearner(state, learning_rate=0.03)
            if method == "bp"
            else CoupledScalingLearner(
                state, learning_rate=0.03, inference_steps=2, inference_learning_rate=0.1
            )
        )
        before = asdict(source.snapshot_state())
        result = run(source, training, heldout, method=method)
        assert result["status"] == "complete" and result["committed_updates"] == 2
        np.testing.assert_equal(asdict(source.snapshot_state()), before)
        assert [row["updates"] for row in result["diagnostics"]] == [0, 1, 2]
        zero = result["diagnostics"][0]
        assert zero["energy"] is zero["task_loss"] is zero["energy_definition"] is None
        assert zero["raw_gradient_norms"] is zero["effective_gradient_norms"] is None
        assert zero["diagnostic_objective_evaluations"] == zero["diagnostic_inference_steps"] == 0
        last = result["diagnostics"][-1]
        final = result["final_state"]
        assert final.config.precisions is not None
        p = effective_coupled_parameters(final)
        np.testing.assert_allclose(
            last["raw_parameter_norms"],
            [np.linalg.norm(a) for a in arrays(final.raw_parameters)],
            rtol=1e-14,
        )
        np.testing.assert_allclose(
            last["effective_parameter_norms"], [np.linalg.norm(a) for a in arrays(p)], rtol=1e-14
        )
        entries = sum(a.size for a in arrays(p))
        assert last["parameter_entries"] == entries
        assert last["native_tensor_bytes"] >= entries * 8
        assert tuple(last["row_ids"]) == training[-1].row_ids and last["training_context"] == "A"
        x, y = training[-1].features, training[-1].targets
        ff = feedforward_latents(p, x)
        np.testing.assert_allclose(
            last["feedforward_layer_rms"], [np.sqrt(np.mean(a * a)) for a in ff], rtol=1e-14
        )
        assert last["logit_rms"] == pytest.approx(
            np.sqrt(np.mean((ff[-1] @ p.output_weight + p.output_bias) ** 2)), rel=1e-14
        )
        if method == "bp":
            expected = evaluate_matched_backprop(p, x, y)
            assert last["energy_definition"] == "numpy_matched_binary_bce_preupdate_v1"
            assert last["energy"] == pytest.approx(expected.loss, abs=1e-14)
            gradients = expected.parameter_gradients
            assert (
                last["diagnostic_objective_evaluations"] == 1
                and last["diagnostic_inference_steps"] == 0
            )
        else:
            z = ff
            for _ in range(2):
                evaluation = evaluate_coupled(p, x, y, z, final.config.precisions)
                z = tuple(a - 0.05 * g for a, g in zip(z, evaluation.latent_gradients, strict=True))
            expected_pc = evaluate_coupled(p, x, y, z, final.config.precisions)
            assert last["energy_definition"] == "numpy_coupled_binary_energy_v1"
            assert last["energy"] == pytest.approx(expected_pc.energy, abs=1e-14)
            gradients = expected_pc.parameter_gradients
            np.testing.assert_allclose(
                last["diagnostic_latent_rms"], [np.sqrt(np.mean(a * a)) for a in z], rtol=1e-14
            )
            assert (
                last["diagnostic_objective_evaluations"] == 3
                and last["diagnostic_inference_steps"] == 2
            )
        np.testing.assert_allclose(
            last["effective_gradient_norms"],
            [np.linalg.norm(a) for a in arrays(gradients)],
            rtol=1e-13,
            atol=1e-15,
        )
        config = final.config
        raw_factors = (
            *config.forward_scales[:-1],
            *((config.bias_scale,) * config.hidden_layers),
            config.forward_scales[-1],
            config.bias_scale,
        )
        np.testing.assert_allclose(
            last["raw_gradient_norms"],
            [
                np.linalg.norm(a * factor)
                for a, factor in zip(arrays(gradients), raw_factors, strict=True)
            ],
            rtol=1e-13,
            atol=1e-15,
        )
        results[method] = result
    for method, result in results.items():
        np.testing.assert_equal(
            asdict(result["initial_state"].raw_parameters),
            asdict(results["pc"]["initial_state"].raw_parameters),
        )
    np.testing.assert_equal(
        asdict(results["pc"]["final_state"].raw_parameters),
        asdict(results["neutral_cpc"]["final_state"].raw_parameters),
    )
