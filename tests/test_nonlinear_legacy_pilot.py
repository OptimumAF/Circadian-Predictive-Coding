"""Legacy Gaussian enumeration, complete seed reporting and tiny runtime controls.

Synthetic summaries and sampler faults never run a pilot. The single real control
uses one training batch per method and one matching manual continuation each.
"""

from copy import deepcopy
from dataclasses import asdict
from itertools import product
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.adapters import nonlinear_scaling_pilot as pilot
from src.app.nonlinear_scaling_measurement import run_nonlinear_scaling_case
from src.app.nonlinear_scaling_summary import (
    ALL_SCHEMES,
    LEGACY_SCHEMES,
    SCHEMES,
    planned_keys,
    summarize_nonlinear_scaling,
)
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.coupled_pc_scaling import effective_coupled_parameters
from src.shared.process_memory import ProcessRssSegment

METHODS = ("bp", "pc", "neutral_cpc", "active_cpc")
SEEDS = (23, 41, 59)
CHECKPOINTS = (0, 1, 2, 3)


def independent_keys(schemes):
    return tuple(product((2, 8), (1, 3), schemes, SEEDS, METHODS))


def complete_rows(schemes):
    rows = []
    for width, depth, scheme, seed, method in independent_keys(schemes):
        value = (
            0.5
            + SEEDS.index(seed) * 0.1
            + (0.2 if method == "bp" else (-0.1 if method == "active_cpc" else 0))
        )
        checkpoints = [
            dict(updates=k, scores={"stationary": dict(bce=value, accuracy=0.5)})
            for k in CHECKPOINTS
        ]
        state = SimpleNamespace(raw_parameters={"fixed": seed})
        rows.append(
            dict(
                width=width,
                depth=depth,
                scheme=scheme,
                seed=seed,
                method=method,
                status="complete",
                reason=None,
                committed_updates=3,
                case_result=dict(
                    stream_result={"checkpoints": checkpoints},
                    initial_state=state,
                    final_state=state,
                    diagnostics=[],
                ),
            )
        )
    return rows


def test_should_preserve_default_axes_and_add_only_exact_legacy_and_combined_groups():
    assert SCHEMES == ("fan_in", "mean_field")
    assert LEGACY_SCHEMES == ("legacy_gaussian",)
    assert ALL_SCHEMES == ("legacy_gaussian", "fan_in", "mean_field")
    assert planned_keys() == planned_keys(schemes=SCHEMES)
    for schemes, count in ((SCHEMES, 96), (LEGACY_SCHEMES, 48), (ALL_SCHEMES, 144)):
        actual = planned_keys(schemes=schemes)
        assert len(actual) == len(set(actual)) == count
        assert set(actual) == set(independent_keys(schemes))


@pytest.mark.parametrize(
    "schemes",
    (
        None,
        [],
        ["legacy_gaussian"],
        (),
        ("fan_in",),
        ("mean_field", "fan_in"),
        ("legacy_gaussian", "fan_in"),
        ("fan_in", "mean_field", "legacy_gaussian"),
        ("legacy_gaussian", "legacy_gaussian"),
    ),
)
def test_should_refuse_arbitrary_reordered_or_non_tuple_scheme_groups(schemes: Any):
    with pytest.raises(ValueError):
        planned_keys(schemes=schemes)
    with pytest.raises(ValueError):
        summarize_nonlinear_scaling(tuple(complete_rows(LEGACY_SCHEMES)), schemes=schemes)


@pytest.mark.parametrize(
    "schemes,count", ((SCHEMES, 448), (LEGACY_SCHEMES, 224), (ALL_SCHEMES, 672))
)
def test_should_report_complete_ordered_seed_vectors_and_prespecified_pairs(schemes, count):
    summaries = summarize_nonlinear_scaling(tuple(complete_rows(schemes)), schemes=schemes)
    assert len(summaries) == count
    assert set(s["scheme"] for s in summaries) == set(schemes)
    for row in summaries:
        assert [o["seed"] for o in row["summary"]["observations"]] == list(SEEDS)
        assert row["summary"]["observed_seed_count"] == 3
    pc = next(s for s in summaries if s["comparison"] == "pc" and s["metric"] == "bce")
    assert pc["summary"]["mean"] == pytest.approx(0.6)
    interval = pc["summary"]["marginal_interval"]
    assert interval["lower"] == pytest.approx(0.6 - 4.303 * 0.1 / np.sqrt(3))
    assert interval["upper"] == pytest.approx(0.6 + 4.303 * 0.1 / np.sqrt(3))
    for comparison, expected in (
        ("pc_minus_bp", -0.2),
        ("active_cpc_minus_neutral_cpc", -0.1),
        ("neutral_cpc_minus_pc", 0.0),
    ):
        pair = next(s for s in summaries if s["comparison"] == comparison and s["metric"] == "bce")
        assert pair["difference_direction"] == "left_minus_right"
        assert pair["summary"]["mean"] == pytest.approx(expected)


@pytest.mark.parametrize("status", ("failed", "not_run"))
def test_should_keep_missing_or_failed_legacy_seed_and_null_affected_combined_estimates(status):
    rows = complete_rows(ALL_SCHEMES)
    damaged = next(
        r
        for r in rows
        if (r["width"], r["depth"], r["scheme"], r["seed"], r["method"])
        == (2, 1, "legacy_gaussian", 41, "pc")
    )
    damaged.update(
        status=status, reason="injected missing seed", case_result=None, committed_updates=0
    )
    summaries = summarize_nonlinear_scaling(tuple(rows), schemes=ALL_SCHEMES)
    affected = [
        s
        for s in summaries
        if (s["width"], s["depth"], s["scheme"]) == (2, 1, "legacy_gaussian")
        and s["comparison"] in ("pc", "pc_minus_bp", "neutral_cpc_minus_pc")
    ]
    assert len(affected) == 24
    for row in affected:
        estimate = row["summary"]
        assert estimate["mean"] is estimate["marginal_interval"] is None
        assert estimate["observed_seed_count"] == 2
        assert [o["seed"] for o in estimate["observations"]] == list(SEEDS)
    assert all(
        s["summary"]["observed_seed_count"] == 3 and s["summary"]["mean"] is not None
        for s in summaries
        if s["scheme"] != "legacy_gaussian"
    )


@pytest.mark.parametrize(
    "mutation", ("missing", "duplicate", "width", "depth", "scheme", "seed", "method", "partial")
)
def test_should_refuse_malformed_legacy_run_axes_or_partial_scored_results(mutation):
    rows = complete_rows(LEGACY_SCHEMES)
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[1] = deepcopy(rows[0])
    elif mutation == "width":
        rows[0]["width"] = True
    elif mutation == "depth":
        rows[0]["depth"] = 2
    elif mutation == "scheme":
        rows[0]["scheme"] = "fan_in"
    elif mutation == "seed":
        rows[0]["seed"] = 1
    elif mutation == "method":
        rows[0]["method"] = "winner"
    else:
        rows[0].update(status="failed", reason="injected")
    with pytest.raises(ValueError):
        summarize_nonlinear_scaling(tuple(rows), schemes=LEGACY_SCHEMES)


def test_should_freeze_legacy_protocol_without_changing_existing_controls_or_resource_caps():
    original = pilot.describe_protocol()
    legacy = pilot.describe_protocol(legacy_only=True)
    assert original["protocol_id"] == "nonlinear_raw_scaling_fixed_pilot_v1"
    assert original["planned_runs"] == 96 and original["planned_training_updates"] == 288
    assert legacy["protocol_id"] == "nonlinear_raw_legacy_extension_v1"
    assert legacy["legacy_gamma0"] == 1.0
    assert legacy["schemes"] == ["legacy_gaussian"] or legacy["schemes"] == LEGACY_SCHEMES
    assert legacy["planned_runs"] == 48 and legacy["planned_training_updates"] == 144
    assert legacy["nominal_training_objective_evaluations"] == 504
    assert legacy["nominal_probe_objective_evaluations"] == 360
    for key in (
        "seeds",
        "widths",
        "hidden_depths",
        "methods",
        "input_dim",
        "precisions",
        "learning_rate",
        "inference_steps",
        "inference_learning_rate",
        "active_cpc_config",
        "neutral_cpc_config",
        "train_rng",
        "final_rng",
        "distribution",
        "target_rule",
        "training_batches",
        "batch_size",
        "heldout_rows",
        "label_delay",
        "checkpoint_updates",
        "primary_metric",
        "secondary_metric",
        "score_release",
    ):
        assert legacy[key] == original[key]
    assert (
        legacy["learning_rate"] == 0.01
        and legacy["inference_steps"] == 2
        and legacy["inference_learning_rate"] == 0.05
    )
    for name, expected in (
        ("wall_seconds", 60),
        ("cpu_threads", 1),
        ("process_memory_bytes", 256 * 1024**2),
        ("storage_bytes", 16 * 1024**2),
        ("training_updates", 144),
    ):
        assert legacy["resources"][name] == expected


@pytest.mark.parametrize("width,depth", ((2, 1), (8, 3)))
def test_should_initialize_owned_exact_legacy_draws_maps_factors_and_four_method_predictions(
    width, depth
):
    learners = [
        pilot.build_learner(width, depth, "legacy_gaussian", 23, method, legacy_only=True)
        for method in METHODS
    ]
    states = [l.snapshot_state() for l in learners]
    expected = asdict(states[0].raw_parameters)
    rng = np.random.default_rng(23)
    shapes = ((2, width), *((width, width),) * (depth - 1), (width, 1))
    for actual, shape in zip(
        (*states[0].raw_parameters.hidden_weights, states[0].raw_parameters.output_weight),
        shapes,
        strict=True,
    ):
        np.testing.assert_array_equal(actual, rng.standard_normal(shape) * 0.5)
    features = pilot.build_data()[0][0].features
    for method, state, learner in zip(METHODS, states, learners, strict=True):
        config = state.config
        assert config.scheme == "legacy_gaussian" and config.gamma0 == 1
        assert config.precisions == (1.0,) * depth
        assert (
            config.forward_scales == config.effective_learning_rate_factors == (1.0,) * (depth + 1)
        )
        assert config.initial_stddevs == (0.5,) * (depth + 1)
        assert (
            config.raw_learning_rate_factor
            == config.bias_scale
            == config.effective_bias_learning_rate_factor
            == 1
        )
        np.testing.assert_equal(asdict(state.raw_parameters), expected)
        np.testing.assert_equal(asdict(effective_coupled_parameters(state)), expected)
        np.testing.assert_array_equal(learner.predict(features), learners[0].predict(features))
        if method in ("bp", "pc"):
            assert config.circadian is None and state.wake_state is None
        else:
            assert config.circadian is not None and state.wake_state is not None
            assert config.circadian.min_plasticity == (1.0 if method == "neutral_cpc" else 0.2)
    states[0].raw_parameters.hidden_weights[0][:] = 99
    assert states[2].wake_state is not None
    states[2].wake_state.chemical_state[:] = 99
    for learner in learners:
        np.testing.assert_equal(asdict(learner.snapshot_state().raw_parameters), expected)


@pytest.mark.parametrize(
    "legacy,scheme", ((True, "fan_in"), (True, "mean_field"), (False, "legacy_gaussian"))
)
def test_should_refuse_factory_scheme_outside_selected_protocol(legacy, scheme):
    with pytest.raises(ValueError):
        pilot.build_learner(2, 1, scheme, 23, "bp", legacy_only=legacy)


@pytest.mark.parametrize("flag", (0, 1, None, np.bool_(True), "legacy"))
def test_should_require_exact_bool_selection_before_pilot_or_factory_work(monkeypatch, flag: Any):
    calls = []
    monkeypatch.setattr(pilot, "ProcessRssSampler", lambda: calls.append("sampler"))
    with pytest.raises(ValueError):
        pilot.describe_protocol(legacy_only=flag)
    with pytest.raises(ValueError):
        pilot.build_learner(2, 1, "legacy_gaussian", 23, "bp", legacy_only=flag)
    with pytest.raises(ValueError):
        pilot.run_pilot(legacy_only=flag)
    with pytest.raises(ValueError):
        pilot.unstarted_pilot_report("injected", 0.0, legacy_only=flag)
    assert calls == []


class FakeSampler:
    def sample(self):
        return 2048

    def __enter__(self):
        return self

    def __exit__(self, *arguments):
        return None

    def snapshot(self):
        return ProcessRssSegment(1, 1024, 2048, 1, 0.05)


def test_should_execute_only_48_fake_legacy_keys_and_preserve_all_summary_groups(monkeypatch):
    indexed = {
        key: row
        for key, row in zip(
            independent_keys(LEGACY_SCHEMES), complete_rows(LEGACY_SCHEMES), strict=True
        )
    }
    calls = []

    def fake_run(key, *args, **kwargs):
        calls.append(key)
        return deepcopy(indexed[key])

    monkeypatch.setattr(pilot, "ProcessRssSampler", FakeSampler)
    monkeypatch.setattr(pilot, "_run_one", fake_run)
    result = pilot.run_pilot(legacy_only=True)
    assert (
        result["status"] == "complete"
        and len(result["runs"]) == 48
        and len(result["summaries"]) == 224
    )
    assert len(calls) == len(set(calls)) == 48 and set(calls) == set(
        independent_keys(LEGACY_SCHEMES)
    )


@pytest.mark.parametrize("phase", ("construction", "enter"))
def test_should_keep_all48_unstarted_rows_and_null_scores_after_one_sampler_startup_failure(
    monkeypatch, phase
):
    calls = []

    class Broken(FakeSampler):
        def __init__(self):
            calls.append("construction")
            if phase == "construction":
                raise RuntimeError("legacy startup fault")

        def __enter__(self):
            calls.append("enter")
            raise RuntimeError("legacy startup fault")

    monkeypatch.setattr(pilot, "ProcessRssSampler", Broken)
    monkeypatch.setattr(
        pilot, "build_learner", lambda *args, **kwargs: pytest.fail("startup must not build models")
    )
    result = pilot.run_pilot(legacy_only=True)
    assert result["status"] == "incomplete" and len(result["runs"]) == 48
    assert all(
        row["status"] == "not_run" and row["committed_updates"] == 0 and row["case_result"] is None
        for row in result["runs"]
    )
    assert len(result["summaries"]) == 224 and all(
        row["summary"]["mean"] is row["summary"]["marginal_interval"] is None
        for row in result["summaries"]
    )
    assert calls == (["construction"] if phase == "construction" else ["construction", "enter"])


@pytest.mark.parametrize("legacy", (False, True))
def test_should_describe_selected_cli_protocol_without_model_work(legacy):
    args = [sys.executable, "-m", "scripts.run_nonlinear_scaling_pilot", "--describe"]
    if legacy:
        args.append("--legacy-only")
    result = subprocess.run(
        args,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert not result.stderr
    report = json.loads(result.stdout)
    assert report["planned_runs"] == (48 if legacy else 96)
    assert report["protocol_id"] == (
        "nonlinear_raw_legacy_extension_v1" if legacy else "nonlinear_raw_scaling_fixed_pilot_v1"
    )


def test_should_apply_one_arrived_legacy_update_per_method_and_match_owned_manual_reference():
    training, heldout = pilot.build_data()
    results = {}
    for method in METHODS:
        source = pilot.build_learner(2, 1, "legacy_gaussian", 23, method, legacy_only=True)
        before = asdict(source.snapshot_state())
        manual = source.fork()
        reference = manual.train_batch(training[0].features.copy(), training[0].targets.copy())
        resource = ToyBudgetSession(ToyExecutionBudget(max_training_updates=1), lambda: 0.0)
        result = run_nonlinear_scaling_case(
            source,
            (training[0],),
            heldout,
            method=method,
            checkpoint_updates=(0, 1),
            label_delay=2,
            budget=resource,
            clock=lambda: 0.0,
        )
        assert (
            result["status"] == "complete"
            and result["committed_updates"] == resource.updates_completed == 1
        )
        np.testing.assert_equal(asdict(result["final_state"]), asdict(manual.snapshot_state()))
        np.testing.assert_equal(asdict(source.snapshot_state()), before)
        receipt = result["stream_result"]["applied_updates"][0]
        assert receipt["arrived_at"] == receipt["applied_at"] == 2
        assert receipt["diagnostic"]["definition"] == reference.definition
        assert receipt["diagnostic"]["value"] == reference.value
        assert len(result["diagnostics"]) == 2 and len(result["stream_result"]["checkpoints"]) == 2
        if method in ("neutral_cpc", "active_cpc"):
            assert result["final_state"].wake_state is not None
            assert result["final_state"].wake_state.wake_updates == 1
        results[method] = result
    np.testing.assert_equal(
        asdict(results["pc"]["final_state"].raw_parameters),
        asdict(results["neutral_cpc"]["final_state"].raw_parameters),
    )
