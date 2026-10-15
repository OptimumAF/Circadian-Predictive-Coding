"""Frozen opt-in timing metadata, strict summaries, and nontraining composition."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import src.adapters.coupled_inference_comparison as pilot
import scripts.run_coupled_inference_comparison as cli
from src.app.coupled_inference_summary import CASE_ORDER, METRICS, summarize_inference_comparison
from src.core.coupled_inference import CoupledInferencePolicy
from src.shared.process_memory import ProcessRssSampler
from test_coupled_inference_comparison import Sampler, complete_rows


def rows():
    result = complete_rows()
    for row in result:
        row["result"].update(
            window_budget_seconds=0.03,
            training_window_seconds=0.0315,
            overrun_seconds=0.0015,
            max_iteration_seconds=0.003,
            observed_max_iteration_seconds=0.002,
        )
    return result


def summary(data):
    return summarize_inference_comparison(
        tuple(data), window_seconds=0.03, max_iteration_seconds=0.003
    )


def test_should_preserve_all_three_seed_vectors_and_directions_for_new_window():
    data = rows()
    saved = deepcopy(data)
    result = summary(data)
    assert data == saved and len(result) == 3 * len(METRICS)
    fixed = next(
        x["summary"] for x in result if x["comparison"] == "fixed" and x["metric"] == "final_bce"
    )
    assert [x["seed"] for x in fixed["observations"]] == [23, 41, 59]
    assert fixed["mean"] == pytest.approx(0.6)
    paired = next(
        x
        for x in result
        if x["comparison"] == "adaptive_minus_fixed" and x["metric"] == "final_bce"
    )
    assert paired["difference_direction"] == "adaptive_minus_fixed"
    assert paired["summary"]["mean"] == pytest.approx(0.02)
    assert fixed["marginal_interval"]["lower"] == pytest.approx(0.6 - 4.303 * 0.1 / np.sqrt(3))


def test_should_keep_default_old_qualification_and_accept_inclusive_new_limits():
    assert summarize_inference_comparison(tuple(complete_rows()))
    with pytest.raises(ValueError):
        summarize_inference_comparison(tuple(rows()))
    data = rows()
    for row in data:
        row["result"].update(
            training_window_seconds=1.1 * 0.03, observed_max_iteration_seconds=0.003
        )
    assert summary(data)


@pytest.mark.parametrize(
    "bad",
    (
        "missing_limit",
        "mismatch",
        "missing_observed",
        "unknown",
        "nan",
        "inf",
        "negative",
        "excess",
        "bool",
        "window",
    ),
)
def test_should_refuse_complete_rows_without_matching_known_qualified_granularity(bad):
    data = rows()
    row = data[0]["result"]
    if bad == "missing_limit":
        del row["max_iteration_seconds"]
    elif bad == "mismatch":
        row["max_iteration_seconds"] = 0.002
    elif bad == "missing_observed":
        del row["observed_max_iteration_seconds"]
    elif bad == "window":
        row["window_budget_seconds"] = 0.01
    else:
        row["observed_max_iteration_seconds"] = {
            "unknown": None,
            "nan": float("nan"),
            "inf": float("inf"),
            "negative": -0.001,
            "excess": 0.0031,
            "bool": True,
        }[bad]
    with pytest.raises(ValueError):
        summary(data)


@pytest.mark.parametrize(
    "options",
    (
        {"window_seconds": True},
        {"window_seconds": 0},
        {"window_seconds": float("nan")},
        {"max_iteration_seconds": True},
        {"max_iteration_seconds": np.float64(0.003)},
        {"max_iteration_seconds": 0},
        {"max_iteration_seconds": float("inf")},
        {"max_iteration_seconds": 0.0031, "window_seconds": 0.03},
    ),
)
def test_should_validate_summary_controls_before_rows(options):
    with pytest.raises(ValueError):
        summarize_inference_comparison(tuple(rows()), **options)


def test_should_retain_failed_seed_positions_and_suppress_incomplete_estimates():
    data = rows()
    data[2].update(status="failed", reason="iteration ceiling")
    data[2]["result"] = dict(status="failed", final_scores=None, initial_scores=None)
    for item in summary(data):
        if item["comparison"] in ("adaptive", "adaptive_minus_fixed"):
            result = item["summary"]
            assert result["mean"] is result["marginal_interval"] is None
            assert result["observations"][1] == dict(
                seed=41, value=None, reason="iteration ceiling"
            )


def test_should_describe_distinct_optin_controls_without_changing_other_science():
    old, new = pilot.describe_protocol(), pilot.describe_protocol(granularity_aware=True)
    assert old["protocol_id"] == "coupled_allocated_time_inference_v1"
    assert new["protocol_id"] == "coupled_allocated_time_inference_granularity_v2"
    assert new["training_seconds"] == 0.03 and new["max_iteration_seconds"] == 0.003
    assert new["maximum_window_seconds"] == 1.1 * 0.03
    assert "max_iteration_seconds" not in old
    for key in set(old) - {
        "protocol_id",
        "training_seconds",
        "maximum_window_seconds",
        "limitations",
    }:
        assert new[key] == old[key]
    assert "10ms" in str(old["limitations"]).replace(" ", "")
    new_limitations = str(new["limitations"]).replace(" ", "")
    assert "30ms" in new_limitations and "3ms" in new_limitations
    assert new["adaptive_policy"] == asdict(CoupledInferencePolicy())


@pytest.mark.parametrize("method", ("fixed", "adaptive"))
def test_should_forward_identical_allocations_and_frozen_native_controls_without_training(
    monkeypatch, method
):
    captured = {}
    monkeypatch.setattr(
        pilot, "ToyBudgetSession", lambda *args: SimpleNamespace(attach_memory=lambda sampler: None)
    )

    def fake(initial, train, final, **options):
        captured.update(options)
        assert train.role == "train" and final.role == "final_test"
        return dict(status="failed", reason="injected", committed_updates=0)

    monkeypatch.setattr(pilot, "run_timed_inference_case", fake)
    initial = pilot.CoupledPredictiveCodingNetwork(2, (8, 8), seed=23)
    result = pilot._run_one(
        23,
        method,
        initial,
        pilot.perf_counter(),
        cast(ProcessRssSampler, Sampler()),
        granularity_aware=True,
    )
    assert result["status"] == "failed"
    assert captured["training_seconds"] == 0.03 and captured["max_iteration_seconds"] == 0.003
    assert captured["max_updates"] == 80 and captured["learning_rate"] == 0.03
    assert captured["inference_steps"] == 6 and captured["inference_learning_rate"] == 0.1
    assert captured["policy"] == (CoupledInferencePolicy() if method == "adaptive" else None)


def test_should_compose_once_in_frozen_order_with_owned_shared_pair_initialization(monkeypatch):
    observed = []
    captured = {}

    def one(seed, method, initial, started, sampler, *, granularity_aware=False):
        assert granularity_aware is True
        observed.append((seed, method, asdict(initial.snapshot_state())))
        initial.restore_state(deepcopy(initial.snapshot_state()))
        initial.snapshot_state().parameters.output_bias[:] = 999
        result = rows()[0]["result"]
        return dict(seed=seed, method=method, status="complete", reason=None, result=result)

    real_summary = summarize_inference_comparison

    def summarize(data, **options):
        captured.update(options)
        return real_summary(data, **options)

    monkeypatch.setattr(pilot, "_run_one", one)
    monkeypatch.setattr(pilot, "summarize_inference_comparison", summarize)
    monkeypatch.setattr(pilot, "ProcessRssSampler", Sampler)
    result = pilot.run_pilot(granularity_aware=True)
    assert result["status"] == "complete"
    assert [(seed, method) for seed, method, _ in observed] == list(CASE_ORDER)
    assert captured == dict(window_seconds=0.03, max_iteration_seconds=0.003)
    for seed in (23, 41, 59):
        states = [state for s, _, state in observed if s == seed]
        np.testing.assert_equal(states[0], states[1])


def test_should_keep_every_planned_row_without_retry_after_composition_failure(monkeypatch):
    calls = []

    def one(seed, method, *args, **kwargs):
        calls.append((seed, method))
        raise RuntimeError("injected granularity startup")

    monkeypatch.setattr(pilot, "_run_one", one)
    monkeypatch.setattr(pilot, "ProcessRssSampler", Sampler)
    report = pilot.run_pilot(granularity_aware=True)
    assert calls == [CASE_ORDER[0]]
    assert report["status"] == "incomplete" and len(report["runs"]) == 6
    assert [(r["seed"], r["method"]) for r in report["runs"]] == list(CASE_ORDER)
    assert all(item["summary"]["mean"] is None for item in report["summaries"])


def test_should_retain_all_unstarted_rows_when_memory_sampler_construction_fails(monkeypatch):
    def unavailable():
        raise RuntimeError("injected sampler constructor")

    monkeypatch.setattr(pilot, "ProcessRssSampler", unavailable)
    monkeypatch.setattr(pilot, "_run_one", lambda *args, **kwargs: pytest.fail("case started"))
    report = pilot.run_pilot(granularity_aware=True)
    assert report["status"] == "incomplete" and report["observed_process_rss"] is None
    assert report["global_errors"] == ["RuntimeError: injected sampler constructor"]
    assert [(row["seed"], row["method"]) for row in report["runs"]] == list(CASE_ORDER)
    assert all(row["status"] == "not_run" for row in report["runs"])
    assert all(item["summary"]["mean"] is None for item in report["summaries"])


@pytest.mark.parametrize("flag", (1, None, "yes", np.bool_(True)))
def test_should_refuse_non_native_bool_protocol_flags_before_model_work(monkeypatch, flag):
    monkeypatch.setattr(
        pilot,
        "CoupledPredictiveCodingNetwork",
        lambda *args, **kwargs: pytest.fail("model construction"),
    )
    for function in (pilot.describe_protocol, pilot.run_pilot):
        with pytest.raises(ValueError):
            function(granularity_aware=flag)
    with pytest.raises(ValueError):
        pilot._run_one(23, "fixed", cast(Any, None), 0.0, cast(Any, None), granularity_aware=flag)


@pytest.mark.parametrize("allowance", (5, 6))
def test_should_charge_crlf_and_terminal_newline_before_printing(monkeypatch, capsys, allowance):
    monkeypatch.setattr(sys, "argv", ["comparison", "--describe", "--granularity-aware"])
    monkeypatch.setattr(cli, "describe_protocol", lambda **kwargs: {})
    monkeypatch.setattr(cli, "MAX_OUTPUT_BYTES", allowance)
    monkeypatch.setattr(cli.os, "linesep", "\r\n")
    monkeypatch.setattr(cli.json, "dumps", lambda *args, **kwargs: "a\nb")
    # LF output plus terminal newline is four bytes; actual CRLF output is six.
    if allowance == 5:
        with pytest.raises(RuntimeError, match="fixed output allowance"):
            cli.main()
        assert capsys.readouterr().out == ""
    else:
        cli.main()
        assert capsys.readouterr().out == "a\nb\n"


def test_should_describe_cli_optin_without_starting_native_training():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.run_coupled_inference_comparison",
            "--describe",
            "--granularity-aware",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )
    assert result.stderr == ""
    assert json.loads(result.stdout) == pilot.describe_protocol(granularity_aware=True)
