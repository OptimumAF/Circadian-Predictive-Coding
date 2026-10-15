"""Synthetic whole-window timing and isolation, plus four real native calls.

Synthetic exact-class models make timing deterministic. The native control uses
an injected clock and does not run the scientific allocated-time comparison.
"""

from copy import deepcopy
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

import src.app.timed_coupled_inference as timed
from src.app.coupled_attribution import AttributionBatch
from src.app.timed_coupled_inference import run_timed_inference_case
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.coupled_energy import CoupledParameters
from src.core.coupled_inference import CoupledInferenceDiagnostics, CoupledInferencePolicy
from src.core.coupled_predictive_coding import (
    CoupledPredictiveCodingNetwork,
    CoupledSnapshot,
    CoupledTrainResult,
)


def test_should_include_terminal_deadline_check_before_window_end(monkeypatch):
    model_type, _, _, _ = fake_model(monkeypatch)
    model_type.duration = 0.0
    stamp = 0.0

    def advancing_clock():
        nonlocal stamp
        stamp += 0.125
        return stamp

    result = run(model_type(), clock=advancing_clock, training_seconds=1.0)
    # Two complete calls, then a terminal deadline check and its closing read.
    # Exact powers of two distinguish the terminal bound without float fuzz.
    assert result["committed_updates"] == 2
    assert result["training_window_seconds"] == 1.25
    assert result["status"] == "failed"
    assert result["initial_scores"] is result["final_scores"] is None


class Clock:
    def __init__(self):
        self.value = 0.0
        self.fail_next = False

    def __call__(self):
        if self.fail_next:
            self.fail_next = False
            raise ValueError("deliberate clock failure")
        return self.value


def batches():
    return (
        AttributionBatch(
            "train",
            "toy",
            ("train-0", "train-1"),
            np.array([[0.2, -0.3], [0.7, 0.1]]),
            np.array([[0.0], [1.0]]),
        ),
        AttributionBatch(
            "final_test",
            "toy",
            ("test-0", "test-1"),
            np.array([[1.1, 0.6], [-0.8, 0.4]]),
            np.ones((2, 1)),
        ),
    )


def make_budget(max_updates=4, clock=lambda: 0.0, wall=None):
    return ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=max_updates, max_wall_seconds=wall), clock
    )


def run(initial, training=None, heldout=None, **options):
    defaults = batches()
    return run_timed_inference_case(
        initial,
        defaults[0] if training is None else training,
        defaults[1] if heldout is None else heldout,
        training_seconds=options.pop("training_seconds", 0.01),
        max_updates=options.pop("max_updates", 4),
        learning_rate=options.pop("learning_rate", 0.03),
        inference_steps=options.pop("inference_steps", 2),
        inference_learning_rate=options.pop("inference_learning_rate", 0.1),
        policy=options.pop("policy", None),
        budget=options.pop("budget", make_budget()),
        **options,
    )


def fake_model(monkeypatch):
    clock = Clock()
    trace: list[tuple[Any, ...]] = []

    class FakePC:
        duration = 0.005
        fail_train = None
        fail_clock_after_train = False
        fallback = False
        fail_predict = None
        prediction = None
        after_predict = None
        total_train_calls = 0
        total_predict_calls = 0

        def __init__(self, input_dim=2, hidden_dims=(2,), seed=0, precisions=None):
            self.input_dim, self.hidden_dims = input_dim, hidden_dims
            self.precisions = (1.0,) * len(hidden_dims) if precisions is None else tuple(precisions)
            dimensions = (input_dim, *hidden_dims)
            p = CoupledParameters(
                tuple(np.zeros((a, b)) for a, b in zip(dimensions, dimensions[1:])),
                tuple(np.zeros((1, b)) for b in hidden_dims),
                np.zeros((hidden_dims[-1], 1)),
                np.zeros((1, 1)),
            )
            self.saved = CoupledSnapshot(1, p, self.precisions)

        def snapshot_parameters(self):
            return deepcopy(self.saved.parameters)

        def restore_parameters(self, p):
            self.saved = replace(self.saved, parameters=deepcopy(p))

        def snapshot_state(self):
            return deepcopy(self.saved)

        def restore_state(self, snapshot):
            self.saved = deepcopy(snapshot)

        def train_epoch(
            self,
            features,
            targets,
            learning_rate,
            inference_steps,
            inference_learning_rate,
            *,
            inference_policy=None,
        ):
            FakePC.total_train_calls += 1
            trace.append(("train", features.copy(), targets.copy(), inference_policy))
            clock.value += FakePC.duration
            if FakePC.total_train_calls == FakePC.fail_train:
                raise RuntimeError("deliberate native failure")
            count = float(self.saved.parameters.output_bias[0, 0]) + 1
            self.saved.parameters.output_bias[0, 0] = count
            if FakePC.fail_clock_after_train:
                clock.fail_next = True
            diagnostic = (
                None
                if inference_policy is None
                else CoupledInferenceDiagnostics(
                    "gradient_tolerance", 1, 1, 2, 2, 1, 10.0, 9.0, 0.0, False
                )
            )
            if inference_policy is not None and FakePC.fallback:
                diagnostic = CoupledInferenceDiagnostics(
                    "energy_increase", 1, 0, 2, 1, 1, 10.0, 10.0, 1.0, True
                )
            return CoupledTrainResult(10.0 - count, inference=diagnostic)

        def predict_proba(self, features):
            FakePC.total_predict_calls += 1
            trace.append(
                ("predict", features.copy(), float(self.saved.parameters.output_bias[0, 0]))
            )
            if FakePC.total_predict_calls == FakePC.fail_predict:
                raise RuntimeError("deliberate score failure")
            features.fill(999)
            if FakePC.after_predict is not None:
                FakePC.after_predict()
            if FakePC.prediction is not None:
                return FakePC.prediction
            return np.full((len(features), 1), 0.25 + 0.1 * self.saved.parameters.output_bias[0, 0])

    monkeypatch.setattr(timed, "CoupledPredictiveCodingNetwork", FakePC)
    return FakePC, FakePC(), clock, trace


@pytest.mark.parametrize("adaptive", (False, True))
def test_should_defer_owned_scores_and_count_actual_fixed_or_adaptive_work(monkeypatch, adaptive):
    model_type, initial, clock, trace = fake_model(monkeypatch)
    training, heldout = batches()
    before = deepcopy((training, heldout))
    result = run(
        initial,
        training,
        heldout,
        clock=clock,
        policy=CoupledInferencePolicy() if adaptive else None,
    )
    assert result["status"] == "complete"
    assert result["attempted_updates"] == result["committed_updates"] == 2
    assert model_type.total_train_calls == 2
    assert [record["updates"] for record in result["records"]] == [1, 2]
    assert [record["energy"] for record in result["records"]] == [9, 8]
    assert all(
        record["energy_definition"] == "numpy_coupled_binary_energy_v1"
        for record in result["records"]
    )
    assert [event[0] for event in trace] == ["train", "train", "predict", "predict"]
    assert [event[2] for event in trace if event[0] == "predict"] == [0, 2]
    for event in trace:
        if event[0] == "train":
            np.testing.assert_array_equal(event[1], training.features)
            np.testing.assert_array_equal(event[2], training.targets)
            assert not np.array_equal(event[2], heldout.targets)
        else:
            np.testing.assert_array_equal(event[1], heldout.features)
    assert initial.saved.parameters.output_bias[0, 0] == 0
    for observed, expected in zip((training, heldout), before, strict=True):
        np.testing.assert_array_equal(observed.features, expected.features)
        np.testing.assert_array_equal(observed.targets, expected.targets)
    assert result["initial_scores"]["bce"] == pytest.approx(-np.log(0.25))
    assert result["final_scores"]["bce"] == pytest.approx(-np.log(0.45))
    assert result["initial_scores"]["accuracy"] == result["final_scores"]["accuracy"] == 0
    assert result["training_window_seconds"] == pytest.approx(0.01)
    assert result["window_budget_seconds"] == 0.01
    assert result["train_call_seconds"] == pytest.approx(0.01)
    assert result["overrun_seconds"] == pytest.approx(0)
    assert result["last_iteration_seconds"] == pytest.approx(0.005)
    work = result["work"]
    assert work["latent_steps"] == (2 if adaptive else 4)
    assert work["inference_energy_evaluations"] == (4 if adaptive else 6)
    assert work["parameter_energy_evaluations"] == 2
    assert work["gradient_norm_evaluations"] == (4 if adaptive else 0)
    assert work["energy_comparisons"] == (2 if adaptive else 0)
    assert work["fallback_updates"] == 0
    assert work["training_row_exposures"] == 4


def test_should_include_budget_and_bookkeeping_in_whole_training_window(monkeypatch):
    model_type, initial, clock, _ = fake_model(monkeypatch)
    model_type.duration = 0.0045
    budget = make_budget()
    before_update, record_update = budget.before_update, budget.record_update

    def charged_before_update():
        before_update()
        clock.value = round(clock.value + 0.00025, 12)

    def charged_record_update():
        record_update()
        clock.value = round(clock.value + 0.00025, 12)

    monkeypatch.setattr(budget, "before_update", charged_before_update)
    monkeypatch.setattr(budget, "record_update", charged_record_update)
    result = run(initial, clock=clock, budget=budget)
    assert result["status"] == "complete"
    assert result["committed_updates"] == 2
    assert result["train_call_seconds"] == pytest.approx(0.009)
    assert result["training_window_seconds"] == pytest.approx(0.01)
    assert result["last_iteration_seconds"] == pytest.approx(0.005)
    assert result["training_window_seconds"] > result["train_call_seconds"]


def test_should_count_attempted_latent_work_and_fallback_updates(monkeypatch):
    model_type, initial, clock, _ = fake_model(monkeypatch)
    model_type.fallback = True
    result = run(initial, clock=clock, policy=CoupledInferencePolicy())
    assert result["status"] == "complete"
    assert result["work"]["latent_steps"] == 2
    assert result["work"]["fallback_updates"] == 2
    assert result["work"]["gradient_norm_evaluations"] == 2
    assert all(record["used_fallback"] for record in result["records"])
    assert all(record["inference"]["accepted_steps"] == 0 for record in result["records"])


@pytest.mark.parametrize(
    "failure",
    (
        "overrun",
        "early_cap",
        "zero_budget",
        "native",
        "clock_after_commit",
        "scoring",
        "last_score_budget",
    ),
)
def test_should_preserve_commits_and_clear_all_scores_on_incomplete_or_failed_case(
    monkeypatch, failure
):
    model_type, initial, clock, _ = fake_model(monkeypatch)
    options: dict[str, Any] = dict(clock=clock)
    expected_commits = 2
    if failure == "overrun":
        model_type.duration = 0.006
    elif failure == "early_cap":
        options["max_updates"] = 1
        expected_commits = 1
    elif failure == "zero_budget":
        options["budget"] = make_budget(0)
        expected_commits = 0
    elif failure == "native":
        model_type.fail_train = 2
        expected_commits = 1
    elif failure == "clock_after_commit":
        model_type.fail_clock_after_train = True
        expected_commits = 1
    elif failure == "scoring":
        model_type.fail_predict = 2
    else:
        wall = [0.0]

        def after_predict():
            if model_type.total_predict_calls == 2:
                wall[0] = 2.0

        model_type.after_predict = after_predict
        options["budget"] = make_budget(clock=lambda: wall[0], wall=1.0)
    result = run(initial, **options)
    assert result["status"] == "failed"
    assert result["reason"]
    assert result["committed_updates"] == expected_commits
    assert result["initial_scores"] is result["final_scores"] is None
    if failure in ("overrun", "early_cap", "zero_budget", "native", "clock_after_commit"):
        assert model_type.total_predict_calls == 0
    if failure == "native":
        assert result["attempted_updates"] == 2
        assert len(result["records"]) == 1
    assert initial.saved.parameters.output_bias[0, 0] == 0


@pytest.mark.parametrize(
    "prediction",
    (
        [[0.5], [0.5]],
        np.ones((2,)),
        np.ones((2, 1), dtype=np.float32),
        np.full((2, 1), np.nan),
        np.full((2, 1), -0.1),
        np.full((2, 1), 1.1),
    ),
)
def test_should_reject_malformed_predictions_without_partial_scores(monkeypatch, prediction):
    model_type, initial, clock, _ = fake_model(monkeypatch)
    model_type.prediction = prediction
    result = run(initial, clock=clock)
    assert result["status"] == "failed"
    assert result["committed_updates"] == 2
    assert result["initial_scores"] is result["final_scores"] is None


def test_should_use_inclusive_half_probability_accuracy(monkeypatch):
    model_type, initial, clock, _ = fake_model(monkeypatch)
    model_type.prediction = np.full((2, 1), 0.5)
    result = run(initial, clock=clock)
    assert result["status"] == "complete"
    assert result["initial_scores"]["accuracy"] == result["final_scores"]["accuracy"] == 1


@pytest.mark.parametrize("owner", ("clock", "budget"))
def test_should_own_model_policy_and_batches_before_external_callbacks(monkeypatch, owner):
    _, initial, clock, trace = fake_model(monkeypatch)
    training, heldout = batches()
    old_x, old_y, old_heldout = (
        training.features.copy(),
        training.targets.copy(),
        heldout.features.copy(),
    )
    configuration = CoupledInferencePolicy()
    armed, fired = [False], [False]

    def callback():
        if armed[0] and not fired[0]:
            fired[0] = True
            initial.saved.parameters.output_bias.fill(99)
            for array in (training.features, training.targets, heldout.features, heldout.targets):
                array.fill(99)
            object.__setattr__(configuration, "gradient_tolerance", 99.0)
        return clock()

    budget = make_budget(clock=callback if owner == "budget" else lambda: 0.0)
    armed[0] = True
    result = run(
        initial,
        training,
        heldout,
        clock=callback if owner == "clock" else clock,
        budget=budget,
        policy=configuration,
    )
    assert fired[0] and result["status"] == "complete"
    assert result["initial_scores"]["bce"] == pytest.approx(-np.log(0.25))
    for event in trace:
        if event[0] == "train":
            np.testing.assert_array_equal(event[1], old_x)
            np.testing.assert_array_equal(event[2], old_y)
            assert event[3].gradient_tolerance == 1e-6
        else:
            np.testing.assert_array_equal(event[1], old_heldout)


@pytest.mark.parametrize(
    "options",
    (
        {"training_seconds": True},
        {"training_seconds": 0},
        {"training_seconds": float("nan")},
        {"max_updates": True},
        {"max_updates": 0},
        {"max_updates": 513},
        {"learning_rate": np.float64(0.03)},
        {"inference_steps": 0},
        {"inference_steps": 10001},
        {"inference_learning_rate": float("inf")},
        {"policy": {}},
        {"budget": None},
        {"clock": None},
    ),
)
def test_should_refuse_invalid_controls_before_training(monkeypatch, options: Any):
    model_type, initial, _, _ = fake_model(monkeypatch)
    with pytest.raises(ValueError):
        run(initial, **options)
    assert model_type.total_train_calls == model_type.total_predict_calls == 0


@pytest.mark.parametrize("failure", ("role", "overlap", "row", "dtype", "family"))
def test_should_refuse_bad_roles_storage_rows_dtype_or_model_family(monkeypatch, failure):
    model_type, initial, _, _ = fake_model(monkeypatch)
    training, heldout = batches()
    if failure == "role":
        heldout = replace(heldout, role="train")
    elif failure == "overlap":
        heldout = replace(heldout, features=training.features)
    elif failure == "row":
        heldout = replace(heldout, row_ids=("train-0", "test-1"))
    elif failure == "dtype":
        training = replace(training, features=training.features.astype(np.float32))
    else:
        initial = object()
    with pytest.raises(ValueError):
        run(initial, training, heldout)
    assert model_type.total_train_calls == model_type.total_predict_calls == 0


def test_should_run_exactly_four_real_native_updates_with_an_injected_clock(monkeypatch):
    original_train = CoupledPredictiveCodingNetwork.train_epoch
    calls = [0]
    clock = Clock()

    def timed_train(self, *args, **kwargs):
        result = original_train(self, *args, **kwargs)
        calls[0] += 1
        clock.value += 0.005
        return result

    monkeypatch.setattr(CoupledPredictiveCodingNetwork, "train_epoch", timed_train)
    for configuration in (None, CoupledInferencePolicy()):
        initial = CoupledPredictiveCodingNetwork(2, (2,), seed=23)
        before = initial.snapshot_parameters()
        clock.value = 0.0
        result = run(initial, clock=clock, policy=configuration)
        assert result["status"] == "complete"
        assert result["committed_updates"] == result["attempted_updates"] == 2
        assert result["training_window_seconds"] == pytest.approx(0.01)
        assert len(result["records"]) == 2
        np.testing.assert_array_equal(
            initial.snapshot_parameters().output_weight, before.output_weight
        )
    assert calls[0] == 4
