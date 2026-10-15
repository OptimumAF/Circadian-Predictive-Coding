"""Rejected detached transforms consume attempts without mutating native owners."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import numpy as np
import pytest

from scripts import run_coupled_rejected_consolidations as cli
from src.adapters import coupled_rejected_consolidations as demo
from src.core.coupled_circadian_predictive_coding import CoupledCircadianPredictiveCodingNetwork
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork


def forbidden(*args, **kwargs):
    pytest.fail("pure description dispatched operational work")


def test_should_describe_fixed_work_without_io_models_threads_or_rss(monkeypatch):
    monkeypatch.setattr(CoupledPredictiveCodingNetwork, "__init__", forbidden)
    monkeypatch.setattr(CoupledCircadianPredictiveCodingNetwork, "__init__", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(threading, "Thread", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(demo, "ProcessRssSampler", forbidden)
    result = demo.describe_example()
    assert result["protocol_id"] == "coupled_rejected_consolidations_engineering_v1"
    assert result["families"] == ["pc", "neutral_cpc", "active_cpc"]
    assert result["native_training_calls"] == 9 and result["native_energy_evaluations"] == 36
    assert result["owned_child_launches"] == 0
    assert "not continuous" in result["limitations"]


def test_should_describe_cli_without_running_example(monkeypatch, capsys):
    monkeypatch.setattr(cli, "run_example", forbidden)
    monkeypatch.setattr(sys, "argv", ["rejected-consolidations", "--describe"])
    cli.main()
    assert json.loads(capsys.readouterr().out) == demo.describe_example()


def test_should_run_cli_once_without_adding_configuration_knobs(monkeypatch, capsys):
    calls = []

    def run():
        calls.append(1)
        return {"fixed": True}

    monkeypatch.setattr(cli, "run_example", run)
    monkeypatch.setattr(sys, "argv", ["rejected-consolidations"])
    cli.main()
    assert calls == [1] and json.loads(capsys.readouterr().out) == {"fixed": True}


@pytest.mark.parametrize("flag", ["--seed", "--count", "--directory"])
def test_should_refuse_unfrozen_cli_configuration_before_run(monkeypatch, flag):
    monkeypatch.setattr(cli, "run_example", forbidden)
    monkeypatch.setattr(sys, "argv", ["rejected-consolidations", flag, "1"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_should_charge_crlf_and_terminal_newline_before_printing(monkeypatch, capsys):
    monkeypatch.setattr(cli, "MAX_OUTPUT_BYTES", 12)
    monkeypatch.setattr(cli.json, "dumps", lambda *args, **kwargs: "x" * 10 + "\n")
    monkeypatch.setattr(cli.os, "linesep", "\r\n")
    # Encoded LF text + terminal LF is12, translated output is14.
    with pytest.raises(RuntimeError):
        cli._print_report({})
    assert capsys.readouterr().out == ""


def test_should_accept_inclusive_output_byte_boundary(monkeypatch, capsys):
    monkeypatch.setattr(cli, "MAX_OUTPUT_BYTES", 12)
    monkeypatch.setattr(cli.json, "dumps", lambda *args, **kwargs: "x" * 11)
    monkeypatch.setattr(cli.os, "linesep", "\n")
    cli._print_report({})
    assert capsys.readouterr().out == "x" * 11 + "\n"


def test_should_refuse_nonfinite_report_instead_of_emitting_nonstandard_json(capsys):
    with pytest.raises(ValueError):
        cli._print_report({"bad": float("nan")})
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("rss", [None, 512 * 1024**2 + 1])
def test_should_refuse_unavailable_or_overbudget_rss_before_constructing_models(monkeypatch, rss):
    monkeypatch.setattr(demo, "read_process_rss_bytes", lambda: rss)
    monkeypatch.setattr(demo, "_model", forbidden)
    with pytest.raises(RuntimeError):
        demo.run_example()


def native_bytes(state):
    parameters = state["parameters"]
    groups = list(parameters["hidden_weights"]) + list(parameters["hidden_biases"])
    groups += [parameters["output_weight"], parameters["output_bias"]]
    wake = state["wake_state"]
    if wake is not None:
        groups += [wake["chemical_state"], wake["plasticity_state"]]
    return sum(np.asarray(value, dtype=np.float64).nbytes for value in groups)


def assert_cursor(cursor, original, expected_ids):
    assert cursor["attempted_ids"] == expected_ids
    assert cursor["consolidations"] == []
    assert cursor["consolidation_limit"] == 8
    assert cursor["stopped"] is False and cursor["retired"] is False
    assert cursor["payload_ready"] is True
    assert cursor["actor_version"] == original["actor_version"]
    assert cursor["learner_version"] == original["learner_version"]


def assert_rss(segment):
    assert type(segment["pid"]) is int and segment["pid"] == os.getpid()
    for name in ("start_bytes", "peak_bytes", "sample_count"):
        assert type(segment[name]) is int and segment[name] > 0
    assert segment["start_bytes"] <= segment["peak_bytes"] <= 512 * 1024**2
    assert np.isfinite(segment["interval_seconds"]) and segment["interval_seconds"] > 0


def test_should_preserve_actor_cache_and_candidate_through_all_rejected_attempts():
    report = demo.run_example()
    assert report["families"] == ["pc", "neutral_cpc", "active_cpc"]
    assert [row["family"] for row in report["rows"]] == report["families"]
    assert 0 <= report["elapsed_seconds"] <= 60
    assert_rss(report["process_rss_segment"])
    assert_rss(report["initial_memory"]["rss_segment"])
    previous_memory = report["initial_memory"]
    states = {}
    for row in report["rows"]:
        actor = row["actor_before"]
        candidate = row["candidate_before"]
        states[row["family"]] = (actor["model_state"], candidate["state"])
        assert row["actor_after"] == actor
        assert actor["model_state"] == row["source_state"]
        after = deepcopy(row["candidate_after"])
        assert candidate["consolidation_attempts"] == candidate["consolidations_completed"] == 0
        assert after.pop("consolidation_attempts") == 8
        expected = deepcopy(candidate)
        expected.pop("consolidation_attempts")
        assert after == expected
        assert candidate["wake_updates"] == 1
        assert candidate["state"]["parameters"] != actor["model_state"]["parameters"]
        assert len(actor["cache"]) == 1
        assert actor["configuration"]["max_cache_entries"] == 2
        cache_entry = next(iter(actor["cache"].values()))
        experience, label = row["experience"], row["label"]
        assert experience["role"] == label["role"] == "train"
        assert experience["sample_id"] == label["sample_id"]
        assert experience["episode_id"] == label["episode_id"]
        assert experience["observed_at"] < label["arrived_at"] <= actor["last_cache_tick"]
        updates = row["applied_updates"]
        assert len(updates) == 1 and updates[0]["update_number"] == 1
        assert updates[0]["event_id"] == label["event_id"]
        assert updates[0]["sample_id"] == experience["sample_id"]
        original = row["cursor_before"]
        assert_cursor(original, original, [])
        ids = []
        steps = row["steps"]
        assert len(steps) == 10
        expected_kinds = ["rejected", "duplicate"] + ["rejected"] * 7 + ["exhausted"]
        assert [step["kind"] for step in steps] == expected_kinds
        calls = 0
        previous_revision = original["revision"]
        for step in steps:
            assert step["callback_count_before"] == calls
            if step["kind"] == "rejected":
                assert step["event_id"] == f"reject-{calls:02d}"
                ids.append(step["event_id"])
                calls += 1
                assert step["refusal"] == "injected detached consolidation refusal"
                mutation = step["mutation"]
                assert mutation["weight_before"] != mutation["weight_after"]
                assert (
                    mutation["native_array_bytes_before"]
                    == mutation["native_array_bytes_after"]
                    == native_bytes(candidate["state"])
                )
                if row["family"] != "pc":
                    assert mutation["chemistry_before"] != mutation["chemistry_after"]
            else:
                assert step["event_id"] == (
                    "reject-00" if step["kind"] == "duplicate" else "reject-08"
                )
                assert step["refusal"]
                assert step["mutation"] is None
            assert step["callback_count_after"] == calls
            assert_cursor(step["cursor"], original, ids)
            assert previous_revision <= step["cursor"]["revision"]
            previous_revision = step["cursor"]["revision"]
            serialized = json.dumps(step["cursor"], allow_nan=False, separators=(",", ":")).encode()
            assert len(serialized) == step["cursor_bytes"] <= 16384
            assert_rss(step["rss_segment"])
            serving = step["serving"]
            assert serving["cache_hit"] is True
            assert serving["generation"] == actor["generation"]
            assert serving["actor_version"] == actor["actor_version"]
            assert serving["metadata"] == actor["metadata"]
            assert serving["prediction"] == cache_entry["prediction"]
        assert calls == 8
        assert row["cursor_after"] == steps[-1]["cursor"]
        assert row["cursor_after"]["revision"] > original["revision"]
        measurements = row["measurements"]
        assert measurements["max_cursor_bytes"] == max(step["cursor_bytes"] for step in steps)
        assert measurements["cache_entries_before"] == measurements["cache_entries_after"] == 1
        expected_bytes = dict(
            actor_before=native_bytes(actor["model_state"]),
            actor_after=native_bytes(actor["model_state"]),
            candidate_before=native_bytes(candidate["state"]),
            candidate_after=native_bytes(candidate["state"]),
        )
        assert measurements["native_array_bytes"] == expected_bytes
        boundaries = measurements["memory_boundaries"]
        assert len(boundaries) >= 2 * len(steps) + 4
        for boundary in boundaries:
            assert type(boundary["phase"]) is str and boundary["phase"]
            assert np.isfinite(boundary["elapsed_seconds"])
            assert (
                previous_memory["elapsed_seconds"]
                <= boundary["elapsed_seconds"]
                <= report["elapsed_seconds"]
            )
            current = boundary["rss_segment"]
            prior = previous_memory["rss_segment"]
            assert_rss(current)
            assert current["start_bytes"] == report["process_rss_segment"]["start_bytes"]
            assert (
                prior["sample_count"]
                <= current["sample_count"]
                <= report["process_rss_segment"]["sample_count"]
            )
            assert (
                prior["peak_bytes"]
                <= current["peak_bytes"]
                <= report["process_rss_segment"]["peak_bytes"]
            )
            previous_memory = boundary
        for step in steps:
            match = next(
                boundary
                for boundary in boundaries
                if boundary["phase"] == f"after_{step['kind']}_{step['event_id']}"
            )
            assert step["rss_segment"] == match["rss_segment"]
        if row["family"] != "pc":
            assert actor["model_state"]["wake_state"]["wake_updates"] == 2
            assert candidate["state"]["wake_state"]["wake_updates"] == 3
    assert states["pc"][0]["parameters"] == states["neutral_cpc"][0]["parameters"]
    assert states["pc"][1]["parameters"] == states["neutral_cpc"][1]["parameters"]
