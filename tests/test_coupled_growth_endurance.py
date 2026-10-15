"""A tiny native prefix validates the fixed endurance engine without a full run."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts import run_coupled_growth_endurance as cli
from src.adapters import coupled_growth_endurance as demo


def forbidden(*args, **kwargs):
    pytest.fail("pure admission/description dispatched operational work")


@pytest.mark.parametrize("value", [True, False, 3, 129, 4.0, np.int64(4), None])
def test_should_refuse_invalid_attempt_limit_before_operational_work(monkeypatch, value):
    monkeypatch.setattr(demo, "monotonic", forbidden)
    monkeypatch.setattr(demo, "ProcessRssSampler", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    with pytest.raises(ValueError):
        demo._run(attempt_limit=value, reads_per_request=2)


@pytest.mark.parametrize("value", [True, False, 1, 65, 2.0, np.int64(2), None])
def test_should_refuse_invalid_read_limit_before_operational_work(monkeypatch, value):
    monkeypatch.setattr(demo, "monotonic", forbidden)
    monkeypatch.setattr(demo, "ProcessRssSampler", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    with pytest.raises(ValueError):
        demo._run(attempt_limit=4, reads_per_request=value)


def test_should_describe_full_work_without_engine_io_rss_or_processes(monkeypatch):
    monkeypatch.setattr(demo, "_run", forbidden)
    monkeypatch.setattr(demo, "ProcessRssSampler", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    result = demo.describe_example()
    assert result["protocol_id"] == "coupled_growth_endurance_engineering_v1"
    assert result["families"] == ["pc", "neutral_cpc", "active_cpc"]
    assert result["native_training_calls"] == 9 and result["native_energy_evaluations"] == 36
    assert result["owned_child_launches"] == result["guarded_inference_calls"] == 0
    assert result["attempt_limit"] == 128 and result["reads_per_request"] == 64
    assert result["serve_calls"] == 24966 and result["cache_hits"] == 24963
    assert result["consolidation_requests"] == 390 and result["split_callbacks"] == 384


def test_should_route_public_run_to_frozen_full_controls_without_running_them(monkeypatch):
    calls = []
    sentinel = {"frozen": True}

    def run(*, attempt_limit, reads_per_request):
        calls.append((attempt_limit, reads_per_request))
        return sentinel

    monkeypatch.setattr(demo, "_run", run)
    assert demo.run_example() is sentinel and calls == [(128, 64)]


def test_should_describe_cli_without_engine_work(monkeypatch, capsys):
    monkeypatch.setattr(cli, "run_example", forbidden)
    monkeypatch.setattr(sys, "argv", ["growth-endurance", "--describe"])
    cli.main()
    assert json.loads(capsys.readouterr().out) == demo.describe_example()


def test_should_route_default_cli_once_without_changing_controls(monkeypatch, capsys):
    calls = []

    def run():
        calls.append(1)
        return {"fixed": True}

    monkeypatch.setattr(cli, "run_example", run)
    monkeypatch.setattr(sys, "argv", ["growth-endurance"])
    cli.main()
    assert calls == [1] and json.loads(capsys.readouterr().out) == {"fixed": True}


@pytest.mark.parametrize("flag", ["--attempt-limit", "--reads-per-request", "--seed"])
def test_should_refuse_cli_workload_knobs_before_native_work(monkeypatch, flag):
    monkeypatch.setattr(cli, "run_example", forbidden)
    monkeypatch.setattr(sys, "argv", ["growth-endurance", flag, "1"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_should_charge_crlf_and_terminal_newline_before_printing(monkeypatch, capsys):
    monkeypatch.setattr(cli, "MAX_OUTPUT_BYTES", 12)
    monkeypatch.setattr(cli.json, "dumps", lambda *args, **kwargs: "x" * 10 + "\n")
    monkeypatch.setattr(cli.os, "linesep", "\r\n")
    with pytest.raises(RuntimeError):
        cli._print_report({})
    assert capsys.readouterr().out == ""


def test_should_accept_inclusive_output_boundary(monkeypatch, capsys):
    monkeypatch.setattr(cli, "MAX_OUTPUT_BYTES", 12)
    monkeypatch.setattr(cli.json, "dumps", lambda *args, **kwargs: "x" * 11)
    monkeypatch.setattr(cli.os, "linesep", "\n")
    cli._print_report({})
    assert capsys.readouterr().out == "x" * 11 + "\n"


def test_should_refuse_nonfinite_json_before_output(capsys):
    with pytest.raises(ValueError):
        cli._print_report({"bad": float("inf")})
    assert capsys.readouterr().out == ""


def test_should_bound_aggregate_summary_by_utf8_bytes_inclusively():
    assert demo._bounded_summary({"x": "a" * (16384 - 8)}) == 16384
    with pytest.raises(ValueError):
        demo._bounded_summary({"x": "a" * (16384 - 7)})
    with pytest.raises(ValueError):
        demo._bounded_summary({"x": "é" * 8192})


def test_should_refuse_nonfinite_summary_values():
    with pytest.raises(ValueError):
        demo._bounded_summary({"bad": float("nan")})


def native_bytes(state):
    p = state["raw_parameters"]
    arrays = (
        list(p["hidden_weights"])
        + list(p["hidden_biases"])
        + [p["output_weight"], p["output_bias"]]
    )
    if state["wake_state"] is not None:
        arrays += [state["wake_state"]["chemical_state"], state["wake_state"]["plasticity_state"]]
    return sum(np.asarray(array, dtype=np.float64).nbytes for array in arrays)


def assert_segment(segment):
    assert segment["pid"] == os.getpid()
    assert 0 < segment["start_bytes"] <= segment["peak_bytes"] <= 512 * 1024**2
    assert type(segment["sample_count"]) is int and segment["sample_count"] > 0
    assert np.isfinite(segment["interval_seconds"]) and segment["interval_seconds"] > 0


def scalar_evidence(value):
    if type(value) is dict:
        assert not set(value).intersection(
            {
                "state",
                "raw_state",
                "parameters",
                "raw_parameters",
                "model_state",
                "candidate",
                "actor",
                "cache",
                "payload_utf8",
                "prediction",
                "predictions",
                "cursor",
                "serving",
            }
        )
        for nested in value.values():
            scalar_evidence(nested)
    elif type(value) is list:
        assert len(value) <= 16
        for nested in value:
            scalar_evidence(nested)
    else:
        assert value is None or type(value) in (int, float, str, bool)
        if type(value) is float:
            assert np.isfinite(value)


def test_should_stream_bounded_evidence_and_continue_after_tiny_native_growth_prefix():
    report = demo._run(attempt_limit=4, reads_per_request=2)
    assert report["run_kind"] == "engineering_control"
    assert [row["family"] for row in report["rows"]] == ["pc", "neutral_cpc", "active_cpc"]
    assert 0 <= report["elapsed_seconds"] <= 240
    assert_segment(report["process_rss_segment"])
    assert report["attempt_limit"] == 4 and report["reads_per_request"] == 2
    expected = dict(
        requests=6,
        unique_attempts=4,
        accepted=2,
        width_refusals=2,
        duplicates=1,
        exhausted=1,
        callbacks=4,
        serve_calls=14,
        cache_hits=13,
        callback_cache_hits=4,
    )
    assert report["totals"] == {key: value * 3 for key, value in expected.items()}
    aggregate = [row["summary"] for row in report["rows"]]
    aggregate_bytes = len(
        json.dumps(aggregate, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )
    assert aggregate_bytes == report["summary_bytes"] <= 16384
    states = {}
    previous_elapsed = 0.0
    previous_samples = 0
    previous_peak = 0
    for row in report["rows"]:
        actor = row["actor_before"]
        assert row["actor_after"] == actor
        assert actor["model_state"]["raw_state"]["config"]["width"] == 2
        assert actor["model_state"]["raw_state"] == row["source_state"]
        assert len(actor["cache"]) == 1 and actor["configuration"]["max_cache_entries"] == 2
        initial, capped, continued = (
            row["candidate_initial"],
            row["candidate_capped"],
            row["candidate_final"],
        )
        states[row["family"]] = (initial, capped, continued)
        assert initial["state"]["raw_state"]["config"]["width"] == 2
        assert (
            capped["state"]["raw_state"]["config"]["width"]
            == continued["state"]["raw_state"]["config"]["width"]
            == 8
        )
        assert (
            initial["state"]["origin_config"]
            == capped["state"]["origin_config"]
            == continued["state"]["origin_config"]
        )
        assert capped["state"]["transitions"] == continued["state"]["transitions"]
        assert [entry["kind"] for entry in capped["state"]["transitions"]] == ["split", "split"]
        assert capped["consolidation_attempts"] == continued["consolidation_attempts"] == 4
        assert capped["consolidations_completed"] == continued["consolidations_completed"] == 2
        assert capped["wake_updates"] == 0 and continued["wake_updates"] == 1
        assert capped["state"]["raw_state"]["config"] == continued["state"]["raw_state"]["config"]
        cursor = row["final_cursor"]
        assert cursor["attempted_ids"] == [f"split-{index:03d}" for index in range(1, 5)]
        assert len(cursor["consolidations"]) == 2
        assert not cursor["stopped"] and not cursor["retired"] and cursor["payload_ready"]
        assert len(json.dumps(cursor, separators=(",", ":")).encode()) <= 16384
        summary = row["summary"]
        assert len(json.dumps(summary, allow_nan=False, separators=(",", ":")).encode()) <= 16384
        scalar_evidence(summary)
        assert summary["counters"] == expected
        assert all(type(value) is int and value >= 0 for value in summary["counters"].values())
        assert (
            "steps" not in row
            and "requests" not in row
            and "reads" not in row
            and "memory_boundaries" not in row
        )
        assert len(summary["blocks"]) <= 8 and len(summary["samples"]) <= 6
        scalar_evidence(summary["blocks"])
        scalar_evidence(summary["samples"])
        for sample in summary["samples"]:
            assert set(sample) == {
                "kind",
                "event_id",
                "requested_width",
                "accepted_width",
                "refusal",
                "callbacks",
                "serve_calls",
            }
            assert all(
                value is None or type(value) in (int, str, bool) for value in sample.values()
            )
        assert [sample["event_id"] for sample in summary["samples"]] == [
            "split-001",
            "split-002",
            "split-003",
            "split-003",
            "split-004",
            "split-005",
        ]
        assert [sample["kind"] for sample in summary["samples"]] == [
            "accepted",
            "accepted",
            "width_refused",
            "duplicate",
            "width_refused",
            "exhausted",
        ]
        assert len(summary["blocks"]) == 1
        assert summary["blocks"][0]["unique_through"] == 4
        block_counts = dict(expected, requests=5, exhausted=0, serve_calls=11, cache_hits=10)
        assert summary["blocks"][0]["counters"] == block_counts
        assert [sample["callbacks"] for sample in summary["samples"]] == [1, 2, 3, 3, 4, 4]
        assert [sample["serve_calls"] for sample in summary["samples"]] == [3, 5, 7, 9, 11, 13]
        assert summary["max_cursor_bytes"] <= 16384
        assert (
            len(json.dumps(cursor, separators=(",", ":")).encode()) <= summary["max_cursor_bytes"]
        )
        assert (
            summary["max_native_array_bytes"]
            == max(
                native_bytes(value["state"]["raw_state"]) for value in (initial, capped, continued)
            )
            <= 968
        )
        assert summary["cache_entries"] == 1
        assert_segment(summary["process_rss_segment"])
        assert (
            summary["process_rss_segment"]["peak_bytes"]
            <= report["process_rss_segment"]["peak_bytes"]
        )
        segment = summary["process_rss_segment"]
        assert segment["start_bytes"] == report["process_rss_segment"]["start_bytes"]
        assert (
            previous_samples
            <= segment["sample_count"]
            <= report["process_rss_segment"]["sample_count"]
        )
        assert previous_peak <= segment["peak_bytes"]
        assert (
            previous_elapsed
            <= summary["start_elapsed_seconds"]
            <= summary["end_elapsed_seconds"]
            <= report["elapsed_seconds"]
        )
        assert 0 < summary["blocks"][0]["rss_peak_bytes"] <= segment["peak_bytes"]
        previous_samples, previous_peak, previous_elapsed = (
            segment["sample_count"],
            segment["peak_bytes"],
            summary["end_elapsed_seconds"],
        )
        assert len(row["applied_updates"]) == 1
        assert row["experience"]["role"] == row["label"]["role"] == "train"
        assert row["experience"]["observed_at"] < row["label"]["arrived_at"]
        if row["family"] != "pc":
            assert initial["state"]["raw_state"]["wake_state"]["wake_updates"] == 2
            assert capped["state"]["raw_state"]["wake_state"]["wake_updates"] == 2
            assert continued["state"]["raw_state"]["wake_state"]["wake_updates"] == 3
        assert native_bytes(continued["state"]["raw_state"]) <= 968
    for phase in range(3):
        assert (
            states["pc"][phase]["state"]["raw_state"]["raw_parameters"]
            == states["neutral_cpc"][phase]["state"]["raw_state"]["raw_parameters"]
        )
