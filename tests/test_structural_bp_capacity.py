"""New structural BP controls through the existing capacity stream/CLI."""

import json
import sys
from typing import Any, cast

import pytest

import scripts.run_stream_rehearsal_pilot as cli
import src.adapters.capacity_pilot as pilot
import src.app.toy_execution_budget as budget_module
from src.shared.process_memory import ProcessRssSegment


def forbidden(*args, **kwargs):
    pytest.fail("unexpected native work")


def test_should_describe_only_new_bp_work_without_constructing_native_models(monkeypatch):
    for name in (
        "build_learner",
        "initialize_coupled_scaling",
        "split_coupled_scaling_width",
        "ProcessRssSampler",
    ):
        monkeypatch.setattr(pilot, name, forbidden)
    old = pilot.describe_protocol()
    new = pilot.describe_bp_protocol()
    assert old["methods"] == ["pc", "neutral_cpc", "gated_cpc"]
    assert old["planned_work"]["native_updates"] == 24
    assert new["methods"] == ["bp"] and new["planned_work"]["native_updates"] == 8
    assert (
        new["planned_work"]["training_rows"] == 16
        and new["planned_work"]["final_score_forwards"] == 12
    )
    assert (
        new["planned_work"]["scheduled_splits"] == new["planned_work"]["fixed_initial_splits"] == 1
    )
    assert "inference_steps" not in new and "inference_learning_rate" not in new
    for key in (
        "seed",
        "scheme",
        "gamma0",
        "training_features",
        "heldout_features",
        "metrics",
        "limits",
        "checkpoint_updates",
        "label_delay",
    ):
        assert new[key] == old[key]
    assert pilot.describe_protocol() == old


@pytest.mark.parametrize("action", ["describe", "run"])
def test_should_route_bp_cli_without_executing_old_capacity_or_rehearsal_cases(
    monkeypatch, capsys, action
):
    monkeypatch.setattr(cli, "describe_capacity_protocol", forbidden)
    monkeypatch.setattr(cli, "run_capacity_pilot", forbidden)
    monkeypatch.setattr(cli, "run_pilot", forbidden)
    monkeypatch.setattr(
        cli,
        "describe_bp_protocol" if action == "describe" else "run_bp_pilot",
        lambda: dict(status="complete", selected=action),
    )
    monkeypatch.setattr(sys, "argv", ["bp", "--scenario", "capacity-bp", f"--{action}"])
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out)["selected"] == action


@pytest.mark.parametrize("flag", ["--recent-only", "--diagnose-learnability"])
def test_should_refuse_other_scenario_flags_before_native_bp(monkeypatch, flag):
    monkeypatch.setattr(cli, "run_bp_pilot", forbidden)
    monkeypatch.setattr(sys, "argv", ["bp", "--scenario", "capacity-bp", "--run", flag])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_should_keep_both_new_arms_unreached_on_sampler_failure(monkeypatch):
    def fail():
        raise RuntimeError("declared BP sampler failure")

    monkeypatch.setattr(pilot, "ProcessRssSampler", fail)
    monkeypatch.setattr(pilot, "build_learner", forbidden)
    report = pilot.run_bp_pilot()
    assert report["status"] == "incomplete" and report["wake_updates"] == 0
    assert len(report["runs"]) == 2 and all(r["status"] == "not_run" for r in report["runs"])
    assert all(r["method"] == "bp" for r in report["runs"])
    assert "declared BP sampler failure" in report["global_errors"][0]


def test_should_return_incomplete_bp_cli_status_without_an_old_case_fallback(monkeypatch, capsys):
    monkeypatch.setattr(cli, "run_bp_pilot", lambda: dict(status="incomplete"))
    monkeypatch.setattr(cli, "run_capacity_pilot", forbidden)
    monkeypatch.setattr(sys, "argv", ["bp", "--scenario", "capacity-bp", "--run"])
    assert cli.main() == 2 and json.loads(capsys.readouterr().out)["status"] == "incomplete"


class FakeSampler:
    start_bytes = 1024

    def sample(self):
        return 1024

    def snapshot(self):
        return ProcessRssSegment(1, 1024, 1024, 1, 0.005)


def test_should_train_native_bp_across_scheduled_split_without_chemical_or_inference_state(
    monkeypatch,
):
    monkeypatch.setattr(budget_module, "ProcessRssSampler", FakeSampler)
    row: dict[str, Any] = dict(
        method="bp", mode="scheduled_split", native_checkpoints=[], checkpoint_reporting_seconds=0.0
    )
    pilot._run_one(row, pilot.monotonic() + 10, cast(Any, FakeSampler()))
    assert row["status"] == "complete", row.get("reason")
    assert row["source_learner_unchanged"] and row["wake_updates"] == 4
    assert [c["width"] for c in row["native_checkpoints"]] == [2, 4, 4]
    assert all(c["native_wake_updates"] is None for c in row["native_checkpoints"])
    cap = row["capacity"]
    assert cap["completed_native_calls"] == 4 and cap["completed_training_rows"] == 8
    assert cap["mean_parameters"] == 13 and cap["peak_parameters"] == 17
    assert cap["mean_total_array_bytes"] == 104 and cap["peak_total_array_bytes"] == 136
    assert cap["mean_auxiliary_bytes"] == cap["peak_auxiliary_bytes"] == 0
    assert cap["split_committed"] == 1 and row["stream_result"]["score_forward_calls"] == 6
    assert all(
        "bce" in r["diagnostic"]["definition"] for r in row["stream_result"]["applied_updates"]
    )
