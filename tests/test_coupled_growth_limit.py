"""Configured structural width ceilings must precede expanded split work."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts import run_coupled_growth_limit as cli
from src.adapters import coupled_growth_limit as demo
import src.app.coupled_structural_consolidation as consolidation
from src.app.coupled_structural_consolidation import CoupledScalingSplitConsolidator
from tests.test_coupled_structural_consolidation import envelope


def forbidden(*args, **kwargs):
    pytest.fail("admission/description reached expanded or operational work")


@pytest.mark.parametrize("value", [True, False, 0, -1, 2**63, 4.0, np.int64(4), "4"])
def test_should_refuse_non_native_or_out_of_range_width_controls(value):
    with pytest.raises(ValueError):
        CoupledScalingSplitConsolidator(max_width=value)


def test_should_keep_default_unbounded_and_expose_readonly_original_ceiling():
    assert CoupledScalingSplitConsolidator().max_width is None
    assert CoupledScalingSplitConsolidator(max_width=None).max_width is None
    operation = CoupledScalingSplitConsolidator(max_width=2**63 - 1)
    assert operation.max_width == 2**63 - 1
    with pytest.raises(AttributeError):
        setattr(operation, "max_width", 2)


@pytest.mark.parametrize("limit", [None, 6])
def test_should_accept_default_and_inclusive_target_width_without_mutating_source(limit):
    source = envelope("active")
    before = asdict(source)
    result = CoupledScalingSplitConsolidator(max_width=limit)(source)
    assert result.state.raw_state.config.width == 6
    assert result.state.raw_state.config.precisions == (0.35, 0.35)
    assert [transition.kind for transition in result.state.transitions] == ["split"]
    np.testing.assert_equal(asdict(source), before)
    assert source.raw_state.wake_state is not None
    assert result.state.raw_state.wake_state is not None
    assert (
        result.state.raw_state.wake_state.wake_updates == source.raw_state.wake_state.wake_updates
    )


def test_should_refuse_target_over_ceiling_before_expanded_kernel_and_preserve_full_source(
    monkeypatch,
):
    source = envelope("active")
    before = asdict(source)
    monkeypatch.setattr(consolidation, "split_coupled_scaling_width", forbidden)
    with pytest.raises(ValueError, match="^split target width exceeds configured ceiling$"):
        CoupledScalingSplitConsolidator(max_width=5)(source)
    np.testing.assert_equal(asdict(source), before)


def test_should_validate_complete_current_state_before_reporting_width_refusal(monkeypatch):
    source = envelope("active")
    object.__setattr__(
        source.raw_state.raw_parameters, "output_bias", np.zeros((1, 1), dtype=np.float32)
    )
    monkeypatch.setattr(consolidation, "split_coupled_scaling_width", forbidden)
    with pytest.raises(ValueError) as error:
        CoupledScalingSplitConsolidator(max_width=5)(source)
    assert str(error.value) != "split target width exceeds configured ceiling"


def test_should_describe_fixed_work_without_io_models_rss_or_processes(monkeypatch):
    monkeypatch.setattr(demo, "initialize_coupled_scaling", forbidden)
    monkeypatch.setattr(demo, "ProcessRssSampler", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    result = demo.describe_example()
    assert result["protocol_id"] == "coupled_growth_limit_engineering_v1"
    assert result["families"] == ["pc", "neutral_cpc", "active_cpc"]
    assert result["native_training_calls"] == 9 and result["native_energy_evaluations"] == 36
    assert result["owned_child_launches"] == result["guarded_inference_calls"] == 0


def test_should_describe_cli_without_executing_native_example(monkeypatch, capsys):
    monkeypatch.setattr(cli, "run_example", forbidden)
    monkeypatch.setattr(sys, "argv", ["growth-limit", "--describe"])
    cli.main()
    assert json.loads(capsys.readouterr().out) == demo.describe_example()


def test_should_dispatch_one_default_cli_example(monkeypatch, capsys):
    calls = []

    def run():
        calls.append(1)
        return {"fixed": True}

    monkeypatch.setattr(cli, "run_example", run)
    monkeypatch.setattr(sys, "argv", ["growth-limit"])
    cli.main()
    assert calls == [1] and json.loads(capsys.readouterr().out) == {"fixed": True}


@pytest.mark.parametrize("flag", ["--width", "--max-width", "--seed"])
def test_should_refuse_unfrozen_cli_configuration_without_native_work(monkeypatch, flag):
    monkeypatch.setattr(cli, "run_example", forbidden)
    monkeypatch.setattr(sys, "argv", ["growth-limit", flag, "1"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_should_include_translated_terminal_newline_in_cli_output_cap(monkeypatch, capsys):
    monkeypatch.setattr(cli, "MAX_OUTPUT_BYTES", 12)
    monkeypatch.setattr(cli.json, "dumps", lambda *args, **kwargs: "x" * 10 + "\n")
    monkeypatch.setattr(cli.os, "linesep", "\r\n")
    with pytest.raises(RuntimeError):
        cli._print_report({})
    assert capsys.readouterr().out == ""


def raw_bytes(state):
    p = state["raw_parameters"]
    arrays = (
        list(p["hidden_weights"])
        + list(p["hidden_biases"])
        + [p["output_weight"], p["output_bias"]]
    )
    if state["wake_state"] is not None:
        arrays += [state["wake_state"]["chemical_state"], state["wake_state"]["plasticity_state"]]
    return sum(np.asarray(array, dtype=np.float64).nbytes for array in arrays)


def assert_rss(segment):
    assert segment["pid"] == os.getpid()
    assert 0 < segment["start_bytes"] <= segment["peak_bytes"] <= 512 * 1024**2
    assert type(segment["sample_count"]) is int and segment["sample_count"] > 0
    assert np.isfinite(segment["interval_seconds"]) and segment["interval_seconds"] > 0


def effective(state):
    p, config = state["raw_parameters"], state["config"]
    width, gamma = config["width"], config["gamma0"]
    scales = [1 / np.sqrt(config["input_dim"]), 1 / np.sqrt(width)]
    bias_scale = 1 / np.sqrt(gamma**2 * width)
    return dict(
        hidden_weights=[
            np.asarray(value) * scale
            for value, scale in zip(p["hidden_weights"], scales, strict=True)
        ],
        hidden_biases=[np.asarray(value) * bias_scale for value in p["hidden_biases"]],
        output_weight=np.asarray(p["output_weight"]) / (gamma * width),
        output_bias=np.asarray(p["output_bias"]) * bias_scale,
    )


def assert_split(before, after):
    old, new = effective(before), effective(after)
    np.testing.assert_allclose(
        new["hidden_weights"][0],
        np.repeat(old["hidden_weights"][0], 2, axis=1),
        rtol=8 * np.finfo(float).eps,
        atol=0,
    )
    np.testing.assert_allclose(
        new["hidden_weights"][1],
        np.repeat(np.repeat(old["hidden_weights"][1] / 2, 2, axis=0), 2, axis=1),
        rtol=8 * np.finfo(float).eps,
        atol=0,
    )
    for a, b in zip(old["hidden_biases"], new["hidden_biases"], strict=True):
        np.testing.assert_allclose(b, np.repeat(a, 2, axis=1), rtol=8 * np.finfo(float).eps, atol=0)
    np.testing.assert_allclose(
        new["output_weight"],
        np.repeat(old["output_weight"] / 2, 2, axis=0),
        rtol=8 * np.finfo(float).eps,
        atol=0,
    )
    np.testing.assert_allclose(
        new["output_bias"], old["output_bias"], rtol=8 * np.finfo(float).eps, atol=0
    )
    assert after["config"]["precisions"] == [value / 2 for value in before["config"]["precisions"]]
    if before["wake_state"] is not None:
        for name in ("chemical_state", "plasticity_state"):
            assert after["wake_state"][name] == np.repeat(before["wake_state"][name], 2).tolist()
        for name in (
            "wake_updates",
            "chemical_decay",
            "chemical_buildup_rate",
            "plasticity_sensitivity",
            "min_plasticity",
        ):
            assert after["wake_state"][name] == before["wake_state"][name]


def test_should_stop_expansion_at_original_ceiling_then_continue_delayed_native_wake():
    report = demo.run_example()
    assert [row["family"] for row in report["rows"]] == ["pc", "neutral_cpc", "active_cpc"]
    assert np.isfinite(report["elapsed_seconds"]) and 0 <= report["elapsed_seconds"] <= 60
    assert_rss(report["process_rss_segment"])
    states = {}
    for row in report["rows"]:
        actor = row["actor_before"]
        assert row["actor_after"] == actor
        assert actor["model_state"]["raw_state"]["config"]["width"] == 2
        assert len(actor["cache"]) == 1
        original = row["candidate_before"]
        before_wake, after_wake = (
            row["candidate_before_continuation"],
            row["candidate_after_continuation"],
        )
        states[row["family"]] = (original, before_wake, after_wake)
        assert original["state"]["raw_state"]["config"]["width"] == 2
        assert before_wake["state"]["raw_state"]["config"]["width"] == 8
        assert after_wake["state"]["raw_state"]["config"]["width"] == 8
        assert (
            before_wake["state"]["origin_config"]
            == after_wake["state"]["origin_config"]
            == original["state"]["origin_config"]
        )
        assert before_wake["state"]["transitions"] == after_wake["state"]["transitions"]
        assert [entry["kind"] for entry in before_wake["state"]["transitions"]] == [
            "split",
            "split",
        ]
        assert before_wake["consolidation_attempts"] == after_wake["consolidation_attempts"] == 3
        assert (
            before_wake["consolidations_completed"] == after_wake["consolidations_completed"] == 2
        )
        assert before_wake["wake_updates"] == 0 and after_wake["wake_updates"] == 1
        steps = row["steps"]
        assert len(steps) == 5
        assert [step["event_id"] for step in steps] == [
            "split-1",
            "split-2",
            "split-3",
            "split-3",
            "split-4",
        ]
        assert [step["kind"] for step in steps] == [
            "accepted",
            "accepted",
            "width_refused",
            "duplicate",
            "exhausted",
        ]
        assert [step["requested_width"] for step in steps] == [4, 8, 16, 16, 16]
        assert [step["accepted_width"] for step in steps] == [4, 8, None, None, None]
        assert steps[2]["refusal"] == "split target width exceeds configured ceiling"
        assert steps[3]["refusal"] and steps[4]["refusal"]
        assert steps[2]["candidate_after"]["state"] == steps[1]["candidate_after"]["state"]
        assert (
            steps[3]["candidate_after"]
            == steps[2]["candidate_after"]
            == steps[4]["candidate_after"]
        )
        assert [step["callback_count_before"] for step in steps] == [0, 1, 2, 3, 3]
        assert [step["callback_count_after"] for step in steps] == [1, 2, 3, 3, 3]
        for step in steps:
            if step["kind"] == "accepted":
                assert_split(
                    step["candidate_before"]["state"]["raw_state"],
                    step["candidate_after"]["state"]["raw_state"],
                )
                assert step["receipt"] is not None and step["refusal"] is None
            else:
                assert step["receipt"] is None
                assert step["native_array_bytes_before"] == step["native_array_bytes_after"]
            serving = step["serving"]
            assert serving["cache_hit"] is True
            assert serving["actor_version"] == actor["actor_version"]
            assert serving["generation"] == actor["generation"]
            assert serving["metadata"] == actor["metadata"]
            assert serving["prediction"] == next(iter(actor["cache"].values()))["prediction"]
            assert (
                len(json.dumps(step["cursor"], separators=(",", ":")).encode())
                == step["cursor_bytes"]
                <= 16384
            )
        cursor = row["cursor_after"]
        assert cursor["attempted_ids"] == ["split-1", "split-2", "split-3"]
        assert len(cursor["consolidations"]) == 2
        assert (
            cursor["stopped"] is False
            and cursor["retired"] is False
            and cursor["payload_ready"] is True
        )
        assert len(row["applied_updates"]) == 1 and row["applied_updates"][0]["update_number"] == 1
        assert row["experience"]["role"] == row["label"]["role"] == "train"
        assert row["experience"]["observed_at"] < row["label"]["arrived_at"]
        assert (
            after_wake["state"]["raw_state"]["config"]
            == before_wake["state"]["raw_state"]["config"]
        )
        assert row["source_state"] == original["state"]["raw_state"]
        measurements = row["measurements"]
        final_cursor_bytes = len(json.dumps(cursor, separators=(",", ":")).encode())
        assert (
            measurements["max_cursor_bytes"]
            == max(final_cursor_bytes, max(step["cursor_bytes"] for step in steps))
            <= 16384
        )
        assert measurements["cache_entries_before"] == measurements["cache_entries_after"] == 1
        assert measurements["native_array_bytes"] == dict(
            actor_before=raw_bytes(actor["model_state"]["raw_state"]),
            actor_after=raw_bytes(actor["model_state"]["raw_state"]),
            candidate_before=raw_bytes(original["state"]["raw_state"]),
            candidate_grown=raw_bytes(before_wake["state"]["raw_state"]),
            candidate_after_continuation=raw_bytes(after_wake["state"]["raw_state"]),
        )
        previous_samples = 0
        previous_elapsed = 0.0
        for boundary in measurements["memory_boundaries"]:
            segment = boundary["rss_segment"]
            assert_rss(segment)
            assert (
                previous_samples
                <= segment["sample_count"]
                <= report["process_rss_segment"]["sample_count"]
            )
            assert previous_elapsed <= boundary["elapsed_seconds"] <= report["elapsed_seconds"]
            assert segment["start_bytes"] == report["process_rss_segment"]["start_bytes"]
            previous_samples, previous_elapsed = (
                segment["sample_count"],
                boundary["elapsed_seconds"],
            )
        if row["family"] != "pc":
            a = original["state"]["raw_state"]["wake_state"]
            b = after_wake["state"]["raw_state"]["wake_state"]
            assert a["wake_updates"] == 2 and b["wake_updates"] == 3
    for phase in range(3):
        assert (
            states["pc"][phase]["state"]["raw_state"]["raw_parameters"]
            == states["neutral_cpc"][phase]["state"]["raw_state"]["raw_parameters"]
        )
