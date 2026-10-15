"""Native cache swaps, rollback and refused operations preserve complete owners."""

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts import run_coupled_cache_transactions as cli
from src.adapters import coupled_cache_transactions as demo
from src.app.promotion_guard_evaluation import PromotionGuardEvaluator
from src.core.actor_ports import CandidateState, VersionedState
from src.core.promotion_guard import GuardBatch, PromotionPolicy
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork
from src.core.coupled_circadian_predictive_coding import CoupledCircadianPredictiveCodingNetwork


def forbidden(*args, **kwargs):
    pytest.fail("pure control reached operational/model/guard work")


def test_should_describe_fixed_controls_without_io_models_rss_or_processes(monkeypatch):
    monkeypatch.setattr(CoupledPredictiveCodingNetwork, "__init__", forbidden)
    monkeypatch.setattr(CoupledCircadianPredictiveCodingNetwork, "__init__", forbidden)
    monkeypatch.setattr(demo, "ProcessRssSampler", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    result = demo.describe_example()
    assert result["protocol_id"] == "coupled_cache_transactions_engineering_v1"
    assert result["families"] == ["pc", "neutral_cpc", "active_cpc"]


def test_should_describe_cli_without_running_example(monkeypatch, capsys):
    monkeypatch.setattr(cli, "run_example", forbidden)
    monkeypatch.setattr(sys, "argv", ["cache-transactions", "--describe"])
    cli.main()
    expected = json.loads(json.dumps(demo.describe_example(), allow_nan=False))
    assert json.loads(capsys.readouterr().out) == expected


def test_should_run_cli_exactly_once_without_added_controls(monkeypatch, capsys):
    calls = []

    def run():
        calls.append(1)
        return {"fixed": True}

    monkeypatch.setattr(cli, "run_example", run)
    monkeypatch.setattr(sys, "argv", ["cache-transactions"])
    cli.main()
    assert calls == [1] and json.loads(capsys.readouterr().out) == {"fixed": True}


@pytest.mark.parametrize("flag", ["--seed", "--count", "--cache-size"])
def test_should_refuse_unfrozen_cli_knobs_before_native_work(monkeypatch, flag):
    monkeypatch.setattr(cli, "run_example", forbidden)
    monkeypatch.setattr(sys, "argv", ["cache-transactions", flag, "1"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_should_charge_output_newlines_before_printing(monkeypatch, capsys):
    monkeypatch.setattr(cli, "MAX_OUTPUT_BYTES", 12)
    monkeypatch.setattr(cli.json, "dumps", lambda *args, **kwargs: "x" * 10 + "\n")
    monkeypatch.setattr(cli.os, "linesep", "\r\n")
    with pytest.raises(RuntimeError):
        cli._print_report({})
    assert capsys.readouterr().out == ""


def test_should_allow_inclusive_output_boundary(monkeypatch, capsys):
    monkeypatch.setattr(cli, "MAX_OUTPUT_BYTES", 12)
    monkeypatch.setattr(cli.json, "dumps", lambda *args, **kwargs: "x" * 11)
    monkeypatch.setattr(cli.os, "linesep", "\n")
    cli._print_report({})
    assert capsys.readouterr().out == "x" * 11 + "\n"


def test_should_refuse_nonfinite_report_before_printing(capsys):
    with pytest.raises(ValueError):
        cli._print_report({"bad": float("nan")})
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("bad", ["final_role", "future_label", "training_overlap"])
def test_should_refuse_guard_metadata_before_touching_native_state_or_payload(bad):
    class Poison:
        def __deepcopy__(self, memo):
            pytest.fail("invalid guard metadata accessed native state or payload")

    policy = PromotionPolicy("one_minus_brier", "canonical_json", 0, -1, 1, 60, 65536, ("0", "1"))
    actor = VersionedState("actor-0", Poison())
    candidate = CandidateState("actor-0", "candidate-0", Poison(), 1, 1, 1)
    new = GuardBatch("new", (("guard-new", "g0"),), 7, Poison(), Poison(), "inner_guard")
    old = GuardBatch("old", (("guard-old", "g0"),), 7, Poison(), Poison(), "inner_guard")
    training_ids = frozenset({("training", "batch-0")})
    if bad == "final_role":
        new = replace(new, role="final_test")
    elif bad == "future_label":
        new = replace(new, labels_arrived_at=8)
    else:
        training_ids = frozenset(new.sample_keys)
    evaluator = PromotionGuardEvaluator(
        build_learner=forbidden,
        utility=forbidden,
        actions=forbidden,
        state_valid=forbidden,
        prediction_valid=forbidden,
        resource_bytes=forbidden,
        state_digest=forbidden,
        clock=forbidden,
    )
    with pytest.raises(ValueError):
        evaluator.evaluate(policy, actor, candidate, new, old, now=7, training_ids=training_ids)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def state_width(serving):
    return serving["model_state"]["raw_state"]["config"]["width"]


def assert_rss(rss):
    assert rss["pid"] == os.getpid()
    assert 0 < rss["start_bytes"] <= rss["peak_bytes"] <= 512 * 1024**2
    assert type(rss["sample_count"]) is int and rss["sample_count"] > 0
    assert np.isfinite(rss["interval_seconds"]) and rss["interval_seconds"] > 0


def test_should_preserve_full_native_cache_transactions_and_refused_operations():
    report = demo.run_example()
    assert report["protocol_id"] == "coupled_cache_transactions_engineering_v1"
    assert [case["family"] for case in report["cases"]] == ["pc", "neutral_cpc", "active_cpc"]
    assert np.isfinite(report["elapsed_seconds"]) and 0 <= report["elapsed_seconds"] <= 60
    assert_rss(report["rss"])
    previous_samples, previous_elapsed, previous_peak = 0, 0.0, 0
    states = {}
    phases = [
        "prime",
        "hit",
        "second",
        "evict",
        "revisit",
        "expire",
        "prepared",
        "committed",
        "promoted_miss",
        "promoted_hit",
        "rolled_back",
        "rollback_hit",
        "rollback_expire",
    ]
    for case in report["cases"]:
        assert case["candidate_after"] == case["candidate_before"]
        assert case["cursor_after"] == case["cursor_before"]
        candidate = case["candidate_before"]
        assert candidate["state"]["raw_state"]["config"]["width"] == 4
        source_config = case["source_state"]["config"]
        target_config = candidate["state"]["raw_state"]["config"]
        assert source_config["scheme"] == target_config["scheme"] == "mean_field"
        assert source_config["input_dim"] == target_config["input_dim"] == 2
        assert source_config["hidden_layers"] == target_config["hidden_layers"] == 2
        assert source_config["gamma0"] == target_config["gamma0"] == 1.5
        assert source_config["precisions"] == [1.0, 1.0] and target_config["precisions"] == [
            0.5,
            0.5,
        ]
        assert source_config["circadian"] == target_config["circadian"]
        assert candidate["state"]["origin_config"] == source_config
        assert (
            candidate["wake_updates"]
            == candidate["consolidation_attempts"]
            == candidate["consolidations_completed"]
            == 1
        )
        assert [entry["kind"] for entry in candidate["state"]["transitions"]] == ["split"]
        assert case["cursor_before"]["attempted_ids"] == ["split-1"]
        assert len(case["cursor_before"]["consolidations"]) == 1
        assert case["cursor_before"]["payload_ready"] is True
        assert (
            case["cursor_before"]["stopped"] is False and case["cursor_before"]["retired"] is False
        )
        observations = case["observations"]
        assert [row["phase"] for row in observations] == phases
        rows = {row["phase"]: row for row in observations}
        snapshots = {name: row["serving"] for name, row in rows.items()}
        original = snapshots["prime"]
        assert original["generation"] == 0 and original["actor_version"] == "actor-0"
        assert state_width(original) == 2
        assert original["model_state"]["raw_state"] == case["source_state"]
        assert original["configuration"] == {"cache_ttl_ticks": 3, "max_cache_entries": 2}
        assert original["metadata"] == {"route": ["original"]}
        expected_ticks = [2, 3, 3, 4, 4, 7, 7, 0, 7, 8, 7, 8, 10]
        assert [snapshots[name]["last_cache_tick"] for name in phases] == expected_ticks
        for name in (
            "prime",
            "hit",
            "second",
            "evict",
            "revisit",
            "expire",
            "prepared",
            "rolled_back",
            "rollback_hit",
            "rollback_expire",
        ):
            assert snapshots[name]["model_state"] == original["model_state"]
            assert snapshots[name]["configuration"] == original["configuration"]
            assert snapshots[name]["metadata"] == original["metadata"]
        hits = dict(
            prime=False,
            hit=True,
            second=False,
            evict=False,
            revisit=False,
            expire=False,
            promoted_miss=False,
            promoted_hit=True,
            rollback_hit=True,
            rollback_expire=False,
        )
        for name, hit in hits.items():
            result = rows[name]["result"]
            snapshot = snapshots[name]
            assert result["cache_hit"] is hit
            assert result["actor_version"] == snapshot["actor_version"]
            assert result["generation"] == snapshot["generation"]
            assert result["metadata"] == snapshot["metadata"]
        for name in ("prepared", "committed", "rolled_back"):
            assert rows[name]["result"] is None
        keys = {name: list(snapshot["cache"]) for name, snapshot in snapshots.items()}
        x1 = keys["prime"][0]
        assert keys["hit"] == [x1]
        assert len(keys["second"]) == 2 and keys["second"][0] == x1
        x2 = keys["second"][1]
        assert keys["evict"][0] == x2 and x1 not in keys["evict"]
        x3 = keys["evict"][1]
        assert keys["revisit"] == [x3, x1]
        assert keys["expire"] == [x1]
        assert snapshots["prime"]["cache"][x1]["expires_at"] == 5
        assert snapshots["hit"]["cache"][x1]["expires_at"] == 5
        assert snapshots["expire"]["cache"][x1]["expires_at"] == 10
        assert snapshots["prepared"] == snapshots["expire"]
        committed = snapshots["committed"]
        assert committed["generation"] == 1 and committed["actor_version"] == "candidate-0"
        assert committed["model_state"] == candidate["state"]
        assert committed["configuration"] == original["configuration"]
        assert committed["cache"] == {} and committed["last_cache_tick"] == 0
        assert committed["metadata"] == {"route": ["promoted"]}
        restored = deepcopy(snapshots["prepared"])
        restored["generation"] = 2
        assert snapshots["rolled_back"] == restored
        assert snapshots["rollback_hit"]["cache"] == restored["cache"]
        assert snapshots["rollback_expire"]["cache"][x1]["expires_at"] == 13
        assert snapshots["rollback_expire"]["last_cache_tick"] == 10
        assert snapshots["promoted_miss"]["cache"][x1]["expires_at"] == 10
        assert snapshots["promoted_hit"]["cache"] == snapshots["promoted_miss"]["cache"]
        receipt = case["receipt"]
        assert receipt == {
            "generation": 1,
            "previous_version": "actor-0",
            "actor_version": "candidate-0",
        }
        guard = case["guard_report"]
        assert guard["policy"] == case["policy"]
        assert case["policy"]["min_new_utility"] == 0
        assert case["policy"]["min_new_gain"] == -1
        assert case["policy"]["max_old_drop"] == 1
        assert case["policy"]["max_prediction_seconds"] == 60
        assert case["policy"]["max_resource_bytes"] == 65536
        assert case["policy"]["allowed_actions"] == ["0", "1"]
        assert guard["decision"] == {"accepted": True, "reasons": []}
        assert guard["new_guard_keys"] == [["guard-new", "g0"]]
        assert guard["old_guard_keys"] == [["guard-old", "g0"]]
        assert guard["label_arrivals"] == [7, 7] and guard["assessed_at"] == 7
        assert set(map(tuple, guard["training_ids"])).isdisjoint(
            {("guard-new", "g0"), ("guard-old", "g0")}
        )
        assert (
            guard["actor_snapshot_digest"] == sha256(canonical(original["model_state"])).hexdigest()
        )
        assert (
            guard["candidate_snapshot_digest"] == sha256(canonical(candidate["state"])).hexdigest()
        )
        assert guard["candidate_revision"] == [1, 1, 1]
        assert (
            guard["evidence"]["candidate_resource_bytes"]
            == len(canonical(candidate["state"]))
            <= 65536
        )
        assert (
            guard["evidence"]["numerically_valid"] is True
            and guard["evidence"]["actions_safe"] is True
        )
        assert case["guard_callback_counts"] == dict(
            state_valid=2, prediction_valid=4, utility=4, actions=4, resource=1
        )
        for name, guard_keys, features, targets in (
            ("new", [["guard-new", "g0"]], [[0.7, -0.4]], [[1.0]]),
            ("old", [["guard-old", "g0"]], [[-0.6, 0.5]], [[0.0]]),
        ):
            batch = case["guards"][name]
            assert batch["role"] == "inner_guard" and batch["labels_arrived_at"] == 7
            assert (
                batch["sample_keys"] == guard_keys
                and batch["features"] == features
                and batch["targets"] == targets
            )
        training = case["training"]
        assert training["experience"]["role"] == training["label"]["role"] == "train"
        assert training["experience"]["observed_at"] < training["label"]["arrived_at"] == 2
        assert len(training["applied_updates"]) == 1
        assert training["applied_updates"][0]["update_number"] == 1
        assert [row["phase"] for row in case["refusals"]] == [
            "backward_tick",
            "final_test_guard",
            "future_guard",
            "training_overlap",
            "consumed_ticket",
            "repeated_rollback",
        ]
        assert all(row["message"] for row in case["refusals"])
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
        states[case["family"]] = (
            case["source_state"],
            candidate["state"]["raw_state"],
            committed["model_state"]["raw_state"],
            restored["model_state"]["raw_state"],
        )
        if case["family"] != "pc":
            assert case["source_state"]["wake_state"]["wake_updates"] == 2
            assert candidate["state"]["raw_state"]["wake_state"]["wake_updates"] == 3
    for phase in range(4):
        assert (
            states["pc"][phase]["raw_parameters"] == states["neutral_cpc"][phase]["raw_parameters"]
        )
