"""One physical native mutation has distinct uncertain and committed fault accounting."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts import run_coupled_native_failures as cli
from src.adapters import coupled_native_failures as demo
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork
from src.core.coupled_circadian_predictive_coding import CoupledCircadianPredictiveCodingNetwork
from src.core.learner_ports import TrainingDiagnostic


def forbidden(*args, **kwargs):
    pytest.fail("pure boundary reached operational native work")


def test_should_describe_native_fault_controls_without_io_models_or_rss(monkeypatch):
    monkeypatch.setattr(CoupledPredictiveCodingNetwork, "__init__", forbidden)
    monkeypatch.setattr(CoupledCircadianPredictiveCodingNetwork, "__init__", forbidden)
    monkeypatch.setattr(demo, "ProcessRssSampler", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    report = demo.describe_example()
    assert report["protocol_id"] == "coupled_native_failures_engineering_v1"
    assert report["families"] == ["pc", "neutral_cpc", "active_cpc"]
    assert report["native_training_calls"] == 18 and report["native_energy_evaluations"] == 72


def test_should_describe_cli_with_json_normalization_without_running(monkeypatch, capsys):
    monkeypatch.setattr(cli, "run_example", forbidden)
    monkeypatch.setattr(sys, "argv", ["native-failures", "--describe"])
    cli.main()
    assert json.loads(capsys.readouterr().out) == json.loads(
        json.dumps(demo.describe_example(), allow_nan=False)
    )


def test_should_run_cli_once_with_fixed_no_argument_interface(monkeypatch, capsys):
    calls = []

    def run():
        calls.append(1)
        return {"fixed": True}

    monkeypatch.setattr(cli, "run_example", run)
    monkeypatch.setattr(sys, "argv", ["native-failures"])
    cli.main()
    assert calls == [1] and json.loads(capsys.readouterr().out) == {"fixed": True}


@pytest.mark.parametrize("flag", ["--seed", "--mode", "--retry"])
def test_should_refuse_unfrozen_cli_knobs_before_native_work(monkeypatch, flag):
    monkeypatch.setattr(cli, "run_example", forbidden)
    monkeypatch.setattr(sys, "argv", ["native-failures", flag, "1"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_should_charge_crlf_and_terminal_newline_before_output(monkeypatch, capsys):
    monkeypatch.setattr(cli, "MAX_OUTPUT_BYTES", 12)
    monkeypatch.setattr(cli.json, "dumps", lambda *args, **kwargs: "x" * 10 + "\n")
    monkeypatch.setattr(cli.os, "linesep", "\r\n")
    with pytest.raises(RuntimeError):
        cli._print_report({})
    assert capsys.readouterr().out == ""


def test_should_accept_inclusive_output_byte_boundary(monkeypatch, capsys):
    monkeypatch.setattr(cli, "MAX_OUTPUT_BYTES", 12)
    monkeypatch.setattr(cli.json, "dumps", lambda *args, **kwargs: "x" * 11)
    monkeypatch.setattr(cli.os, "linesep", "\n")
    cli._print_report({})
    assert capsys.readouterr().out == "x" * 11 + "\n"


def test_should_refuse_nonfinite_json_before_output(capsys):
    with pytest.raises(ValueError):
        cli._print_report({"bad": float("nan")})
    assert capsys.readouterr().out == ""


class Delegate:
    def __init__(self, value=0):
        self.state = {"value": [value]}
        self.calls = []

    def fork(self):
        result = Delegate()
        result.state = deepcopy(self.state)
        return result

    def train_batch(self, features, targets):
        self.calls.append((deepcopy(features), deepcopy(targets)))
        self.state["value"][0] += 1
        return TrainingDiagnostic("fake_native", float(self.state["value"][0]))

    def predict(self, features):
        return np.array([[float(self.state["value"][0])]])

    def snapshot_state(self):
        return deepcopy(self.state)

    def restore_state(self, state):
        self.state = deepcopy(state)


@pytest.mark.parametrize("mode", ["uncertain", "committed"])
def test_should_execute_one_delegate_mutation_with_owned_same_mode_forks(mode):
    delegate = Delegate()
    wrapper = demo._FaultLearner(delegate, mode)
    fork = wrapper.fork()
    features, targets = np.array([[0.2]]), np.array([[0.8]])
    before = wrapper.snapshot_state()
    if mode == "uncertain":
        with pytest.raises(RuntimeError, match="^injected native failure after update$"):
            wrapper.train_batch(features, targets)
    else:
        assert wrapper.train_batch(features, targets) == TrainingDiagnostic("fake_native", 1.0)
    assert len(delegate.calls) == 1
    np.testing.assert_array_equal(delegate.calls[0][0], features)
    np.testing.assert_array_equal(delegate.calls[0][1], targets)
    assert delegate.state == {"value": [1]}
    assert wrapper.train_calls == 1
    assert fork.snapshot_state() == before and fork.train_calls == 0
    snapshot = wrapper.snapshot_state()
    snapshot["value"][0] = 9
    assert wrapper.predict(features).item() == 1
    wrapper.restore_state({"value": [2]})
    assert wrapper.predict(features).item() == 2 and fork.predict(features).item() == 0
    assert wrapper.train_calls == 1
    if mode == "uncertain":
        with pytest.raises(RuntimeError, match="^injected native failure after update$"):
            fork.train_batch(features, targets)
    else:
        assert fork.train_batch(features, targets) == TrainingDiagnostic("fake_native", 1.0)
    assert (
        fork.train_calls == 1
        and fork.predict(features).item() == 1
        and wrapper.predict(features).item() == 2
    )


def assert_rss(rss):
    assert rss["pid"] == os.getpid()
    assert 0 < rss["start_bytes"] <= rss["peak_bytes"] <= 512 * 1024**2
    assert type(rss["sample_count"]) is int and rss["sample_count"] > 0
    assert np.isfinite(rss["interval_seconds"]) and rss["interval_seconds"] > 0


def test_should_preserve_physical_native_mutation_and_truthful_fault_boundary_accounting():
    report = demo.run_example()
    assert report["protocol_id"] == "coupled_native_failures_engineering_v1"
    expected_order = [
        (family, mode)
        for family in ("pc", "neutral_cpc", "active_cpc")
        for mode in ("uncertain", "committed")
    ]
    assert [(case["family"], case["mode"]) for case in report["cases"]] == expected_order
    assert np.isfinite(report["elapsed_seconds"]) and 0 <= report["elapsed_seconds"] <= 60
    assert_rss(report["rss"])
    states = {}
    complete_after = {}
    previous_samples, previous_elapsed, previous_peak = 0, 0.0, 0
    for case in report["cases"]:
        committed = case["mode"] == "committed"
        before, after = case["native_before"], case["native_after"]
        complete_after[(case["family"], case["mode"])] = after
        states[(case["family"], case["mode"])] = (
            case["source_state"],
            before["raw_state"],
            after["raw_state"],
        )
        assert before["raw_state"]["config"]["width"] == after["raw_state"]["config"]["width"] == 4
        assert before["raw_state"]["config"] == after["raw_state"]["config"]
        source_config = case["source_state"]["config"]
        config = after["raw_state"]["config"]
        assert source_config["scheme"] == config["scheme"] == "mean_field"
        assert source_config["input_dim"] == config["input_dim"] == 2
        assert source_config["hidden_layers"] == config["hidden_layers"] == 2
        assert source_config["gamma0"] == config["gamma0"] == 1.5
        assert source_config["precisions"] == [1.0, 1.0] and config["precisions"] == [0.5, 0.5]
        assert source_config["circadian"] == config["circadian"]
        assert before["origin_config"] == after["origin_config"] == case["source_state"]["config"]
        assert before["transitions"] == after["transitions"]
        assert [entry["kind"] for entry in before["transitions"]] == ["split"]
        assert before["raw_state"]["raw_parameters"] != after["raw_state"]["raw_parameters"]
        assert case["actor_before"] == case["actor_after"]
        actor = case["actor_before"]
        assert actor["model_state"]["raw_state"] == case["source_state"]
        assert actor["model_state"]["raw_state"]["config"]["width"] == 2
        assert actor["generation"] == 0 and actor["actor_version"] == "actor-0"
        assert actor["configuration"] == {"cache_ttl_ticks": 10, "max_cache_entries": 2}
        assert actor["metadata"] == {"route": ["original"]}
        assert actor["last_cache_tick"] == 2 and len(actor["cache"]) == 1
        assert next(iter(actor["cache"].values()))["expires_at"] == 12
        events = case["events"]
        stages = (
            ["started", "completed", "committed_failure"] if committed else ["started", "uncertain"]
        )
        assert [event["stage"] for event in events] == stages
        assert events[0]["native_state"] == before and events[0]["completed_updates"] == 0
        assert events[0]["receipt"] is None
        for event in events:
            assert event["source"]["features"] == [[0.2, -0.3], [-0.4, 0.6]]
            assert event["label"]["targets"] == [[1.0], [0.0]]
            assert event["learner_version"] == "candidate-0"
            assert event["source"]["model_version"] == "actor-0"
            assert event["source"]["observed_at"] == 0
            assert event["source"]["role"] == event["label"]["role"] == "train"
            assert (
                event["label"]["model_version"] == "actor-0" and event["label"]["arrived_at"] == 2
            )
            assert event["source"]["sample_id"] == event["label"]["sample_id"] == "s0"
            assert (
                event["source"]["episode_id"] == event["label"]["episode_id"] == "native-failures"
            )
        for event in events[1:]:
            assert event["native_state"] == after
            assert event["completed_updates"] == int(committed)
            assert (event["receipt"] is not None) is committed
        assert case["updates_completed"] == int(committed)
        assert len(case["applied_updates"]) == int(committed)
        if committed:
            receipt = case["applied_updates"][0]
            assert all(event["receipt"] == receipt for event in events[1:])
            assert (
                receipt["actor_version"],
                receipt["learner_version"],
                receipt["observed_at"],
                receipt["arrived_at"],
                receipt["applied_at"],
                receipt["update_number"],
            ) == ("actor-0", "candidate-0", 0, 2, 2, 1)
        assert case["inbox_after"]["stopped"] is True
        assert (
            case["inbox_before"]["stopped"] is False
            and case["inbox_before"]["completed_updates"] == 0
        )
        assert case["inbox_after"]["capacity"] == case["inbox_before"]["capacity"] == 1
        assert case["inbox_after"]["completed_updates"] == int(committed)
        assert case["inbox_after"]["applied"] == case["applied_updates"]
        assert len(case["inbox_after"]["experiences"]) == len(case["inbox_after"]["labels"]) == 1
        assert case["inbox_after"]["experiences"] == case["inbox_before"]["experiences"]
        assert case["inbox_after"]["labels"] == case["inbox_before"]["labels"]
        cursor = case["cursor_after"]
        assert case["cursor_before"]["revision"] == 3 and cursor["revision"] == 4
        assert cursor["attempted_ids"] == ["split-1"] and len(cursor["consolidations"]) == 1
        assert cursor["consolidation_limit"] == 2
        assert (
            cursor["stopped"] is True
            and cursor["retired"] is False
            and cursor["payload_ready"] is True
        )
        expected_cursor = dict(deepcopy(case["cursor_before"]), revision=4, stopped=True)
        assert cursor == expected_cursor
        assert case["poison_copy_attempts"] == case["consolidation_callback_calls"] == 0
        assert case["budget_limits"]["max_training_updates"] == 1
        assert case["train_calls"] == {"candidate": 1, "actor": 0}
        identity = case["budget_identity"]
        assert identity["original"] == identity["runtime"] == identity["inbox"]
        assert identity["original_limits"] == identity["current_limits"]
        serves = case["serves"]
        assert [serve["phase"] for serve in serves] == ["prime"] + [
            f"event_{stage}" for stage in stages
        ] + ["after_failure", "after_refusals"]
        prediction = next(iter(actor["cache"].values()))["prediction"]
        for index, serve in enumerate(serves):
            assert serve["serving"] == actor
            assert serve["result"]["cache_hit"] is (index != 0)
            assert serve["result"]["prediction"] == prediction
            assert (
                serve["result"]["actor_version"] == "actor-0" and serve["result"]["generation"] == 0
            )
            assert serve["result"]["metadata"] == actor["metadata"]
        assert [entry["phase"] for entry in case["refusals"]] == [
            "retry_train",
            "consolidate",
            "public_snapshot",
            "record_label",
        ]
        assert all(entry["message"] for entry in case["refusals"])
        message = (
            "injected observer failure after commit"
            if committed
            else "injected native failure after update"
        )
        assert case["exception"] == {"type": "RuntimeError", "message": message}
        for boundary in case["rss_boundaries"]:
            rss = boundary["rss"]
            assert_rss(rss)
            assert rss["start_bytes"] == report["rss"]["start_bytes"]
            assert previous_samples <= rss["sample_count"] <= report["rss"]["sample_count"]
            assert previous_peak <= rss["peak_bytes"] <= report["rss"]["peak_bytes"]
            assert previous_elapsed <= boundary["elapsed_seconds"] <= report["elapsed_seconds"]
            previous_samples, previous_peak, previous_elapsed = (
                rss["sample_count"],
                rss["peak_bytes"],
                boundary["elapsed_seconds"],
            )
        if case["family"] != "pc":
            assert case["source_state"]["wake_state"]["wake_updates"] == 2
            assert before["raw_state"]["wake_state"]["wake_updates"] == 2
            assert after["raw_state"]["wake_state"]["wake_updates"] == 3
    for family in ("pc", "neutral_cpc", "active_cpc"):
        assert complete_after[(family, "uncertain")] == complete_after[(family, "committed")]
    for mode in ("uncertain", "committed"):
        for phase in range(3):
            assert (
                states[("pc", mode)][phase]["raw_parameters"]
                == states[("neutral_cpc", mode)][phase]["raw_parameters"]
            )
