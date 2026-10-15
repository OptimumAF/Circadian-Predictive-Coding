"""Producing-version feedback must survive promotion without renewing update work."""

from copy import deepcopy
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pytest

from scripts import run_coupled_delayed_feedback as cli
from src.adapters import coupled_delayed_feedback as demo
from src.app.experience_inbox import ExperienceInbox
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork
from src.core.coupled_circadian_predictive_coding import CoupledCircadianPredictiveCodingNetwork
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from tests.test_experience_inbox import RecordingLearner


def forbidden(*args, **kwargs):
    pytest.fail("pure boundary dispatched operational work")


def test_should_describe_fixed_chronology_without_io_native_models_or_rss(monkeypatch):
    monkeypatch.setattr(CoupledPredictiveCodingNetwork, "__init__", forbidden)
    monkeypatch.setattr(CoupledCircadianPredictiveCodingNetwork, "__init__", forbidden)
    monkeypatch.setattr(demo, "ProcessRssSampler", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    result = demo.describe_example()
    assert result["protocol_id"] == "coupled_delayed_feedback_engineering_v1"
    assert result["families"] == ["pc", "neutral_cpc", "active_cpc"]
    assert result["native_training_calls"] == 9 and result["native_energy_evaluations"] == 36


def test_should_describe_cli_using_json_normalized_values_without_running(monkeypatch, capsys):
    monkeypatch.setattr(cli, "run_example", forbidden)
    monkeypatch.setattr(sys, "argv", ["delayed-feedback", "--describe"])
    cli.main()
    expected = json.loads(json.dumps(demo.describe_example(), allow_nan=False))
    assert json.loads(capsys.readouterr().out) == expected


def test_should_run_default_cli_once_without_extra_controls(monkeypatch, capsys):
    calls = []

    def run():
        calls.append(1)
        return {"fixed": True}

    monkeypatch.setattr(cli, "run_example", run)
    monkeypatch.setattr(sys, "argv", ["delayed-feedback"])
    cli.main()
    assert calls == [1] and json.loads(capsys.readouterr().out) == {"fixed": True}


@pytest.mark.parametrize("flag", ["--seed", "--arrival", "--updates"])
def test_should_refuse_unfrozen_cli_controls_before_work(monkeypatch, flag):
    monkeypatch.setattr(cli, "run_example", forbidden)
    monkeypatch.setattr(sys, "argv", ["delayed-feedback", flag, "1"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_should_charge_translated_and_terminal_newlines_before_output(monkeypatch, capsys):
    monkeypatch.setattr(cli, "MAX_OUTPUT_BYTES", 12)
    monkeypatch.setattr(cli.json, "dumps", lambda *args, **kwargs: "x" * 10 + "\n")
    monkeypatch.setattr(cli.os, "linesep", "\r\n")
    with pytest.raises(RuntimeError):
        cli._print_report({})
    assert capsys.readouterr().out == ""


def test_should_accept_exact_output_byte_boundary(monkeypatch, capsys):
    monkeypatch.setattr(cli, "MAX_OUTPUT_BYTES", 12)
    monkeypatch.setattr(cli.json, "dumps", lambda *args, **kwargs: "x" * 11)
    monkeypatch.setattr(cli.os, "linesep", "\n")
    cli._print_report({})
    assert capsys.readouterr().out == "x" * 11 + "\n"


def test_should_refuse_nonfinite_output_before_printing(capsys):
    with pytest.raises(ValueError):
        cli._print_report({"bad": float("nan")})
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("bad", ["role", "version", "chronology", "event_reuse", "duplicate_label"])
def test_should_refuse_poisoned_label_before_copy_and_preserve_complete_inbox(bad):
    copies = []

    class Poison:
        def __deepcopy__(self, memo):
            copies.append(1)
            pytest.fail("refused label target copied")

    learner = RecordingLearner()
    clock = LogicalClock(2)
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=1), lambda: 0.0)
    inbox = ExperienceInbox(
        learner, clock=clock, budget=budget, learner_version="candidate-0", max_experiences=1
    )
    source = Experience(
        "s0",
        "delayed-feedback",
        2,
        "actor-0",
        ["features"],
        "train",
        ExperiencePermissions(training=True),
    )
    target: LabelArrival[Any] = LabelArrival(
        "label-0", "s0", "delayed-feedback", 10, "actor-0", ["targets"]
    )
    inbox.record_experience(source)
    invalid = replace(target, event_id="bad", targets=Poison())
    if bad == "role":
        invalid = replace(invalid, role="final_test")
    elif bad == "version":
        invalid = replace(invalid, model_version="candidate-0")
    elif bad == "chronology":
        invalid = replace(invalid, arrived_at=1)
    else:
        inbox.record_label(target)
        if bad == "event_reuse":
            invalid = replace(invalid, event_id=target.event_id, sample_id="foreign")
        else:
            clock.advance_to(10)
            assert len(inbox.drain()) == 1
    before = asdict(inbox.capture_cursor())
    calls, completed = deepcopy(learner.calls), budget.updates_completed
    with pytest.raises(ValueError) as error:
        inbox.record_label(invalid)
    assert copies == []
    assert asdict(inbox.capture_cursor()) == before
    assert learner.calls == calls and budget.updates_completed == completed
    if bad == "event_reuse":
        assert "duplicate" in str(error.value) and "capacity" not in str(error.value)


def assert_rss(rss):
    assert rss["pid"] == os.getpid()
    assert 0 < rss["start_bytes"] <= rss["peak_bytes"] <= 512 * 1024**2
    assert type(rss["sample_count"]) is int and rss["sample_count"] > 0
    assert np.isfinite(rss["interval_seconds"]) and rss["interval_seconds"] > 0


def test_should_apply_original_producer_feedback_once_across_promotion_and_rollback():
    report = demo.run_example()
    assert report["protocol_id"] == "coupled_delayed_feedback_engineering_v1"
    assert [case["family"] for case in report["cases"]] == ["pc", "neutral_cpc", "active_cpc"]
    assert np.isfinite(report["elapsed_seconds"]) and 0 <= report["elapsed_seconds"] <= 60
    assert_rss(report["rss"])
    phases = [
        "prime",
        "role_refused",
        "version_refused",
        "chronology_refused",
        "waiting_without_label",
        "source_hit",
        "waiting_at_promotion",
        "promoted",
        "promoted_prime",
        "current_version_refused",
        "buffered",
        "event_reused_refused",
        "waiting_buffered",
        "waiting_before_arrival",
        "applied_at_arrival",
        "promoted_hit",
        "repeated_drain",
        "duplicate_label_refused",
        "rolled_back",
        "rollback_hit",
        "post_rollback_drain",
    ]
    revisions = [2, 3, 4, 5, 6, 6, 7, 7, 7, 8, 9, 10, 11, 12, 13, 13, 14, 15, 15, 15, 16]
    states = {}
    previous_samples, previous_elapsed, previous_peak = 0, 0.0, 0
    for case in report["cases"]:
        rows = case["observations"]
        assert [row["phase"] for row in rows] == phases
        assert [row["cursor"]["revision"] for row in rows] == revisions
        by_phase = {row["phase"]: row for row in rows}
        a = by_phase["prime"]
        b = by_phase["promoted_prime"]
        c = by_phase["applied_at_arrival"]
        states[case["family"]] = (
            case["source_state"],
            a["candidate"]["state"]["raw_state"],
            c["candidate"]["state"]["raw_state"],
            b["serving"]["model_state"]["raw_state"],
            by_phase["rolled_back"]["serving"]["model_state"]["raw_state"],
        )
        assert a["serving"]["model_state"]["raw_state"] == case["source_state"]
        assert a["serving"]["model_state"]["raw_state"]["config"]["width"] == 2
        candidate_b = a["candidate"]
        assert candidate_b["state"]["raw_state"]["config"]["width"] == 4
        assert candidate_b["wake_updates"] == 0
        assert candidate_b["consolidation_attempts"] == candidate_b["consolidations_completed"] == 1
        assert [entry["kind"] for entry in candidate_b["state"]["transitions"]] == ["split"]
        source_config = case["source_state"]["config"]
        candidate_config = candidate_b["state"]["raw_state"]["config"]
        assert source_config["scheme"] == candidate_config["scheme"] == "mean_field"
        assert source_config["gamma0"] == candidate_config["gamma0"] == 1.5
        assert source_config["input_dim"] == candidate_config["input_dim"] == 2
        assert source_config["hidden_layers"] == candidate_config["hidden_layers"] == 2
        assert source_config["precisions"] == [1.0, 1.0] and candidate_config["precisions"] == [
            0.5,
            0.5,
        ]
        assert source_config["circadian"] == candidate_config["circadian"]
        assert candidate_b["state"]["origin_config"] == source_config
        assert a["serving"]["configuration"] == {"cache_ttl_ticks": 20, "max_cache_entries": 2}
        assert a["serving"]["metadata"] == {"route": ["original"]}
        source = a["inbox"]["experiences"][0]
        assert a["inbox"]["capacity"] == 1 and a["inbox"]["learner_version"] == "candidate-0"
        assert source["model_version"] == "actor-0" and source["role"] == "train"
        assert source["sample_id"] == "s0" and source["episode_id"] == "delayed-feedback"
        assert source["observed_at"] == 2
        assert a["inbox"]["labels"] == [] and a["inbox"]["applied"] == []
        for index, row in enumerate(rows):
            assert row["updates_completed"] == (0 if index < 14 else 1)
            assert row["candidate"] == (candidate_b if index < 14 else c["candidate"])
            assert row["cursor"]["attempted_ids"] == ["split-1"]
            assert len(row["cursor"]["consolidations"]) == 1
            assert row["cursor"]["payload_ready"] is True
            assert row["cursor"]["stopped"] is False and row["cursor"]["retired"] is False
            assert row["inbox"]["stopped"] is False
            assert row["inbox"]["completed_updates"] == row["updates_completed"]
            assert row["inbox"]["experiences"] == [source]
        assert by_phase["buffered"]["serving"]["actor_version"] == "candidate-0"
        label = by_phase["buffered"]["inbox"]["labels"][0]
        assert label["event_id"] == "label-0" and label["model_version"] == "actor-0"
        assert label["arrived_at"] == 10 and label["targets"] == [[1.0], [0.0]]
        for phase in (
            "buffered",
            "event_reused_refused",
            "waiting_buffered",
            "waiting_before_arrival",
        ):
            assert by_phase[phase]["inbox"]["labels"] == [label]
            assert by_phase[phase]["inbox"]["applied"] == []
        assert c["serving"] == by_phase["waiting_before_arrival"]["serving"]
        assert c["candidate"]["wake_updates"] == 1
        assert (
            c["candidate"]["state"]["raw_state"]["config"]
            == candidate_b["state"]["raw_state"]["config"]
        )
        assert c["candidate"]["state"]["origin_config"] == candidate_b["state"]["origin_config"]
        assert c["candidate"]["state"]["transitions"] == candidate_b["state"]["transitions"]
        assert (
            c["candidate"]["state"]["raw_state"]["raw_parameters"]
            != candidate_b["state"]["raw_state"]["raw_parameters"]
        )
        applied = c["inbox"]["applied"]
        assert len(applied) == 1
        receipt = applied[0]
        assert (
            receipt["actor_version"],
            receipt["learner_version"],
            receipt["sample_id"],
            receipt["episode_id"],
            receipt["event_id"],
        ) == ("actor-0", "candidate-0", "s0", "delayed-feedback", "label-0")
        assert (
            receipt["observed_at"],
            receipt["arrived_at"],
            receipt["applied_at"],
            receipt["update_number"],
        ) == (2, 10, 10, 1)
        for row in rows[14:]:
            assert row["inbox"]["applied"] == applied
            assert row["inbox"]["labels"] == [label]
        committed = by_phase["promoted"]["serving"]
        assert committed["generation"] == 1 and committed["actor_version"] == "candidate-0"
        assert committed["model_state"] == candidate_b["state"]
        assert committed["metadata"] == {"route": ["promoted"]}
        assert committed["cache"] == {} and committed["last_cache_tick"] == 0
        assert committed["configuration"] == a["serving"]["configuration"]
        saved = deepcopy(by_phase["source_hit"]["serving"])
        saved["generation"] = 2
        assert by_phase["rolled_back"]["serving"] == saved
        x1 = next(iter(a["serving"]["cache"]))
        assert a["serving"]["cache"][x1]["expires_at"] == 22
        assert b["serving"]["cache"][x1]["expires_at"] == 27
        for phase, hit in (
            ("prime", False),
            ("source_hit", True),
            ("promoted_prime", False),
            ("waiting_before_arrival", True),
            ("promoted_hit", True),
            ("rollback_hit", True),
        ):
            result = by_phase[phase]["result"]
            assert result["cache_hit"] is hit
            assert result["actor_version"] == by_phase[phase]["serving"]["actor_version"]
        refusals = case["refusals"]
        refusal_phases = [
            "role_refused",
            "version_refused",
            "chronology_refused",
            "current_version_refused",
            "event_reused_refused",
            "duplicate_label_refused",
        ]
        assert [refusal["phase"] for refusal in refusals] == refusal_phases
        for refusal in refusals:
            index = phases.index(refusal["phase"])
            previous, current = rows[index - 1], rows[index]
            assert refusal["revision_before"] == previous["cursor"]["revision"]
            assert (
                refusal["revision_after"]
                == current["cursor"]["revision"]
                == refusal["revision_before"] + 1
            )
            assert refusal["message"]
            for key in ("serving", "candidate", "inbox", "updates_completed"):
                assert previous[key] == current[key]
            expected_cursor = deepcopy(previous["cursor"])
            expected_cursor["revision"] += 1
            assert current["cursor"] == expected_cursor
        assert case["poison_copy_attempts"] == 0
        assert case["budget_limits"]["max_training_updates"] == 1
        identity = case["budget_identity"]
        assert type(identity["original"]) is int
        assert identity["original"] == identity["runtime"] == identity["inbox"]
        assert identity["original_limits"] == identity["current_limits"]
        assert case["guard_callback_counts"] == dict(
            state_valid=2, prediction_valid=4, utility=4, actions=4, resource=1
        )
        guard = case["guard_report"]
        assert guard["actor_version"] == "actor-0" and guard["learner_version"] == "candidate-0"
        assert guard["candidate_revision"] == [0, 1, 1]
        assert guard["training_ids"] == [["delayed-feedback", "s0"]]
        assert guard["label_arrivals"] == [7, 7] and guard["assessed_at"] == 7
        assert guard["new_guard_keys"] == [["guard-new", "g0"]]
        assert guard["old_guard_keys"] == [["guard-old", "g0"]]
        assert guard["decision"] == {"accepted": True, "reasons": []}
        assert case["receipt"] == {
            "generation": 1,
            "previous_version": "actor-0",
            "actor_version": "candidate-0",
        }
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
            assert candidate_b["state"]["raw_state"]["wake_state"]["wake_updates"] == 2
            assert c["candidate"]["state"]["raw_state"]["wake_state"]["wake_updates"] == 3
    for phase_index in range(5):
        assert (
            states["pc"][phase_index]["raw_parameters"]
            == states["neutral_cpc"][phase_index]["raw_parameters"]
        )
