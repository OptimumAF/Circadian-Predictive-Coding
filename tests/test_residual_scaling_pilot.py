"""Residual factory, shared grid and CLI failures; no training steps."""

import json
import subprocess
import sys
from typing import Any

import numpy as np
import pytest

from src.adapters import linear_scaling_pilot as shared
from src.adapters import residual_scaling_pilot as residual
from src.app.linear_scaling_summary import KEYS
from src.core.residual_linear_pc import LinearResidualState
from src.shared.process_memory import ProcessRssSampler


def row() -> dict[str, Any]:
    return dict(
        seed=23,
        width=8,
        hidden_layers=2,
        scheme="mean_field",
        status="not_run",
        reason="not reached",
        attempted_updates=0,
        committed_updates=0,
        checkpoints=[],
        heldout_half_mse=None,
    )


def test_should_keep_original_grid_data_rate_and_metric_without_selection():
    protocol = residual.describe_protocol()
    old = shared.describe_protocol()
    for key in (
        "seeds",
        "widths",
        "hidden_layers",
        "schemes",
        "order",
        "updates",
        "learning_rate",
        "training_rng",
        "heldout_rng",
        "training_rows",
        "heldout_rows",
        "teacher",
        "metric",
        "resources",
    ):
        assert protocol[key] == old[key]
    assert protocol["protocol_id"] == "residual_linear_scaling_measurements_v1"
    assert protocol["topology"] == "fixed_middle_identity"
    assert protocol["seeds"] == [23, 41, 59] and protocol["updates"] == 6
    assert protocol["learning_rate"] == 0.01 and protocol["training_rng"] == 7311
    assert protocol["heldout_rng"] == 9111 and protocol["resources"]["training_updates"] == 486


def test_should_use_exact_residual_factory_and_preserve_partial_train_work(monkeypatch):
    observed = []

    def failed(initial, training, heldout, **options):
        assert type(initial) is LinearResidualState
        assert options["updates"] == 6 and options["learning_rate"] == 0.01
        assert initial.config.depth == 3
        observed.append(initial)
        assert not np.shares_memory(training.features, heldout.features)
        return dict(
            status="failed",
            reason="injected numeric failure",
            attempted_updates=3,
            committed_updates=2,
            checkpoints=[{"updates": i} for i in range(3)],
            heldout_half_mse=None,
        )

    monkeypatch.setattr(residual, "run_residual_scaling_case", failed)
    result = row()
    with ProcessRssSampler(read_rss_bytes=lambda: 1000) as sampler:
        residual._run_one(result, shared.monotonic(), sampler)
    assert len(observed) == 1 and result["status"] == "failed"
    assert result["attempted_updates"] == 3 and result["committed_updates"] == 2
    assert result["heldout_half_mse"] is None
    assert result["work"]["completed_step_gradient_evaluations"] == 4
    assert result["work"]["heldout_forwards"] is None
    assert result["initial_state"]["identity_layer_indices_zero_based"] == [1]
    assert result["initial_state"]["depth"] == 3
    raw = result["initial_state"]["raw_weights"]
    effective = result["initial_state"]["effective_weights"]
    scales = result["initial_state"]["forward_scales"]
    np.testing.assert_allclose(scales, [1 / np.sqrt(2), 1 / np.sqrt(24), 1 / 8])
    for i, (w, a, s) in enumerate(zip(raw, effective, scales)):
        expected = np.array(w) * s + (np.eye(8) if i == 1 else 0)
        np.testing.assert_array_equal(expected, a)
    assert result["case_elapsed_seconds"] >= result["initialization_seconds"] >= 0


def test_should_keep_all_rows_and_missing_seeds_after_each_case_refusal(monkeypatch):
    calls = []
    monkeypatch.setattr(
        shared, "ProcessRssSampler", lambda: ProcessRssSampler(read_rss_bytes=lambda: 1000)
    )

    def refuse(record, started, sampler):
        calls.append((record["seed"], record["width"], record["hidden_layers"], record["scheme"]))
        record.update(status="failed", reason="injected refusal")

    monkeypatch.setattr(residual, "_run_one", refuse)
    report = residual.run_pilot()
    assert tuple(calls) == KEYS and len(report["runs"]) == 81
    assert report["status"] == "incomplete" and report["within_resources"]
    assert report["protocol"]["topology"] == "fixed_middle_identity"
    assert len(report["summaries"]) == 27
    assert all(s["final"]["mean"] is None for s in report["summaries"])
    assert report["work"]["attempted_step_calls"] == 0


def test_should_keep_unreached_rows_on_shared_memory_failure(monkeypatch):
    def fail():
        raise OSError("injected memory failure")

    monkeypatch.setattr(shared, "ProcessRssSampler", lambda: ProcessRssSampler(read_rss_bytes=fail))
    report = residual.run_pilot()
    assert report["status"] == "incomplete" and not report["within_resources"]
    assert len(report["runs"]) == 81 and all(r["status"] == "not_run" for r in report["runs"])
    assert report["global_errors"] and report["observed_process_rss"] is None


def test_should_record_factory_error_without_calling_training(monkeypatch):
    def fail(*args):
        raise ValueError("injected factory failure")

    monkeypatch.setattr(residual, "initialize_residual_linear", fail)
    monkeypatch.setattr(
        residual, "run_residual_scaling_case", lambda *a, **k: pytest.fail("unexpected training")
    )
    record = row()
    with ProcessRssSampler(read_rss_bytes=lambda: 1000) as sampler:
        residual._run_one(record, shared.monotonic(), sampler)
    assert record["status"] == "failed" and "factory failure" in record["reason"]
    assert record["attempted_updates"] == 0 and record["heldout_half_mse"] is None


def test_should_report_shared_wall_exhaustion_without_training(monkeypatch):
    monkeypatch.setattr(shared, "monotonic", lambda: 1000.0)
    monkeypatch.setattr(
        residual, "initialize_residual_linear", lambda *a: pytest.fail("unexpected factory")
    )
    record = row()
    with ProcessRssSampler(read_rss_bytes=lambda: 1000) as sampler:
        residual._run_one(record, 0.0, sampler)
    assert record["status"] == "not_run" and "wall budget" in record["reason"]
    assert record["heldout_half_mse"] is None


def test_should_describe_residual_cli_without_training():
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.run_residual_scaling_pilot", "--describe"],
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    )
    assert completed.stderr == ""
    assert json.loads(completed.stdout) == residual.describe_protocol()


def test_should_emit_incomplete_cli_result_with_exit_two(monkeypatch, capsys):
    from scripts import run_residual_scaling_pilot as cli

    monkeypatch.setattr(sys, "argv", ["residual-scaling"])
    monkeypatch.setattr(
        cli, "run_pilot", lambda: {"status": "incomplete", "runs": [{"status": "not_run"}]}
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert json.loads(capsys.readouterr().out)["runs"] == [{"status": "not_run"}]
