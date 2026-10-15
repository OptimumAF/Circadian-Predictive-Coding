"""Frozen data, composition, complete-vector aggregation and describe CLI controls.

No test in this module executes real training or the scientific timing pilot.
"""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

import src.adapters.coupled_inference_comparison as pilot
from src.app.coupled_inference_summary import (
    CASE_ORDER,
    METRICS,
    WORK_METRICS,
    summarize_inference_comparison,
)
from src.core.coupled_inference import CoupledInferencePolicy
from src.shared.process_memory import ProcessRssSampler


def complete_result(value=0.5):
    return dict(
        status="complete",
        reason=None,
        committed_updates=2,
        attempted_updates=2,
        window_budget_seconds=0.01,
        training_window_seconds=0.0105,
        train_call_seconds=0.009,
        overrun_seconds=0.0005,
        initial_scores=dict(bce=0.7, accuracy=0.5),
        final_scores=dict(bce=value, accuracy=0.75),
        work={name: 2 for name in WORK_METRICS},
    )


def complete_rows():
    return [
        dict(
            seed=seed,
            method=method,
            status="complete",
            reason=None,
            result=complete_result(
                0.5 + 0.1 * (23, 41, 59).index(seed) + (0.02 if method == "adaptive" else 0)
            ),
        )
        for seed, method in CASE_ORDER
    ]


def test_should_build_frozen_independent_role_batches():
    train, final = pilot.build_batches()
    again, _ = pilot.build_batches()
    assert train.role == "train" and final.role == "final_test"
    assert train.features.shape == (16, 2) and final.features.shape == (128, 2)
    assert set(train.row_ids).isdisjoint(final.row_ids)
    np.testing.assert_array_equal(
        train.features, np.random.default_rng(7421).uniform(-1, 1, (16, 2))
    )
    np.testing.assert_array_equal(
        final.features, np.random.default_rng(9221).uniform(-1, 1, (128, 2))
    )
    np.testing.assert_array_equal(train.features, again.features)
    for batch in (train, final):
        np.testing.assert_array_equal(
            batch.targets, batch.features[:, [0]] + 0.25 * batch.features[:, [1]] >= 0
        )
    assert not np.shares_memory(train.features, final.features)
    assert not np.shares_memory(train.targets, final.targets)
    train.features.fill(999)
    assert not np.any(again.features == 999)


def test_should_preserve_all_seed_intervals_and_paired_direction():
    rows = complete_rows()
    saved = deepcopy(rows)
    summaries = summarize_inference_comparison(tuple(rows))
    assert rows == saved and len(summaries) == 3 * len(METRICS)
    fixed = next(
        x["summary"] for x in summaries if x["comparison"] == "fixed" and x["metric"] == "final_bce"
    )
    assert [o["seed"] for o in fixed["observations"]] == [23, 41, 59]
    assert fixed["mean"] == pytest.approx(0.6)
    assert fixed["marginal_interval"]["lower"] == pytest.approx(0.6 - 4.303 * 0.1 / np.sqrt(3))
    paired = next(
        x
        for x in summaries
        if x["comparison"] == "adaptive_minus_fixed" and x["metric"] == "final_bce"
    )
    assert paired["difference_direction"] == "adaptive_minus_fixed"
    assert paired["summary"]["mean"] == pytest.approx(0.02)
    constant = next(x["summary"] for x in summaries if x["metric"] == "committed_updates")
    assert (
        constant["marginal_interval"] is None
        and constant["interval_status"] == "zero_observed_variance"
    )


def test_should_retain_failure_and_suppress_complete_vector_estimates():
    rows = complete_rows()
    rows[2].update(status="failed", reason="overrun")
    rows[2]["result"] = dict(status="failed", final_scores=None, initial_scores=None)
    summaries = summarize_inference_comparison(tuple(rows))
    for item in summaries:
        if item["comparison"] in ("adaptive", "adaptive_minus_fixed"):
            summary = item["summary"]
            assert summary["planned_seed_count"] == 3 and summary["observed_seed_count"] == 2
            assert summary["mean"] is None and summary["marginal_interval"] is None
            assert summary["observations"][1] == dict(seed=41, value=None, reason="overrun")


@pytest.mark.parametrize(
    "mutation",
    [
        "lost",
        "duplicate",
        "reorder",
        "status",
        "window",
        "updates",
        "nan",
        "accuracy",
        "missing",
        "reason",
    ],
)
def test_should_refuse_malformed_or_nonqualified_complete_cases(mutation):
    rows = complete_rows()
    if mutation == "lost":
        rows.pop()
    elif mutation == "duplicate":
        rows[1] = rows[0]
    elif mutation == "reorder":
        rows[0], rows[1] = rows[1], rows[0]
    elif mutation == "status":
        rows[0]["status"] = "best"
    elif mutation == "window":
        rows[0]["result"]["training_window_seconds"] = 0.0111
    elif mutation == "updates":
        rows[0]["result"]["committed_updates"] = True
    elif mutation == "nan":
        rows[0]["result"]["final_scores"]["bce"] = float("nan")
    elif mutation == "accuracy":
        rows[0]["result"]["final_scores"]["accuracy"] = 1.1
    elif mutation == "missing":
        del rows[0]["result"]["work"]["latent_steps"]
    else:
        rows[0].update(status="failed", reason="")
    with pytest.raises(ValueError):
        summarize_inference_comparison(tuple(rows))


class Sampler:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def snapshot(self):
        # Adapter uses dataclasses.asdict; reuse the actual immutable record.
        from src.shared.process_memory import ProcessRssSegment

        return ProcessRssSegment(1000, 1000, 1000, 1, 0.01)


def test_should_compose_once_in_fixed_order_with_exact_owned_initialization(monkeypatch):
    observed = []

    def case(seed, method, initial, started, sampler):
        observed.append((seed, method, asdict(initial.snapshot_state())))
        return dict(
            seed=seed, method=method, status="complete", reason=None, result=complete_result()
        )

    monkeypatch.setattr(pilot, "_run_one", case)
    monkeypatch.setattr(pilot, "ProcessRssSampler", Sampler)
    report = pilot.run_pilot()
    assert [(seed, method) for seed, method, _ in observed] == list(CASE_ORDER)
    assert len(report["runs"]) == 6 and report["status"] == "complete"
    for seed in (23, 41, 59):
        states = [state for s, _, state in observed if s == seed]
        np.testing.assert_equal(states[0], states[1])
    assert set(report["initial_parameters"]) == {"23", "41", "59"}


def test_should_keep_unreached_rows_on_global_failure(monkeypatch):
    count = 0

    def case(seed, method, *args):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("injected outer failure")
        return dict(
            seed=seed, method=method, status="complete", reason=None, result=complete_result()
        )

    monkeypatch.setattr(pilot, "_run_one", case)
    monkeypatch.setattr(pilot, "ProcessRssSampler", Sampler)
    report = pilot.run_pilot()
    assert report["status"] == "incomplete" and len(report["runs"]) == 6 and count == 2
    assert report["runs"][1]["status"] == "not_run"
    assert report["global_errors"] == ["RuntimeError: injected outer failure"]
    assert (
        next(x["summary"] for x in report["summaries"] if x["comparison"] == "adaptive")["mean"]
        is None
    )


@pytest.mark.parametrize("method", ["fixed", "adaptive"])
def test_should_pass_frozen_controls_and_default_policy_without_training(monkeypatch, method):
    captured = {}
    monkeypatch.setattr(
        pilot, "ToyBudgetSession", lambda *args: SimpleNamespace(attach_memory=lambda sampler: None)
    )

    def case(initial, train, final, **kwargs):
        captured.update(kwargs)
        assert train.role == "train" and final.role == "final_test"
        return dict(status="failed", reason="injected unscored", committed_updates=0)

    monkeypatch.setattr(pilot, "run_timed_inference_case", case)
    source = pilot.CoupledPredictiveCodingNetwork(2, (8, 8), 23)
    row = pilot._run_one(
        23, method, source, pilot.perf_counter(), cast(ProcessRssSampler, Sampler())
    )
    assert row["status"] == "failed" and row["result"]["committed_updates"] == 0
    assert captured["training_seconds"] == 0.01 and captured["max_updates"] == 80
    assert (
        captured["learning_rate"] == 0.03
        and captured["inference_steps"] == 6
        and captured["inference_learning_rate"] == 0.1
    )
    assert captured["policy"] == (CoupledInferencePolicy() if method == "adaptive" else None)


def test_should_describe_in_subprocess_without_pilot_training():
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.run_coupled_inference_comparison", "--describe"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        timeout=30,
        check=True,
    )
    report = json.loads(completed.stdout)
    assert report == pilot.describe_protocol() and completed.stderr == ""
    assert report["adaptive_policy"] == asdict(CoupledInferencePolicy())
    assert report["resources"]["training_updates"] == 480
