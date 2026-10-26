"""Existing fake learners and actual toy packets expose reserved diagnostics only."""

from copy import deepcopy
import json
import sys

import numpy as np
import pytest

import scripts.run_coupled_attribution as cli
import src.adapters.key_door_pilot as pilot
import src.app.fresh_task_probe as probe_module
from test_recurrent_key_door_pilot import Learner, fake_environment
from test_torch_feature_rehearsal_pilot import forbidden


def normalize(value):
    """Ignore measured durations, retaining prospective limits and all work."""
    if type(value) is dict:
        return {
            key: normalize(item)
            for key, item in value.items()
            if key not in ("total_seconds", "elapsed_seconds")
        }
    if type(value) is list:
        return [normalize(item) for item in value]
    return value


def without_decisions(report, original_protocol):
    result = deepcopy(report)
    result["protocol"] = deepcopy(original_protocol)
    for row in result["runs"]:
        for cohort in row["probes"]:
            cohort["report"].pop("decision_diagnostics_enabled")
            cohort["report"].pop("decision_diagnostics_scope")
            for point in cohort["report"]["checkpoints"]:
                point.pop("decision_diagnostics")
    return result


def test_should_change_only_opt_in_protocol_metadata_and_preserve_default_descriptor(monkeypatch):
    for name in (
        "ProcessRssSampler",
        "require_torch",
        "build_stream",
        "build_probe_stream",
        "run_fresh_task_probe",
        "KeyDoorNavigation",
    ):
        monkeypatch.setattr(pilot, name, forbidden)
    ordinary = pilot.describe_learnability_protocol()
    assert ordinary == pilot.describe_learnability_protocol(decision_diagnostics=False)
    enabled = pilot.describe_learnability_protocol(decision_diagnostics=True)
    assert ordinary["protocol_id"] == "actual_key_door_owned_checkpoint_learnability_v1"
    assert enabled["protocol_id"] == "actual_key_door_owned_checkpoint_decisions_v1"
    assert enabled["probe"]["decision_diagnostics"] is True
    assert enabled["probe"]["decision_diagnostics_scope"]
    adjusted = deepcopy(enabled)
    adjusted["protocol_id"], adjusted["probe"] = ordinary["protocol_id"], ordinary["probe"]
    assert adjusted == ordinary


def test_should_forward_only_requested_flag_to_all_four_probes_without_changing_old_results(
    monkeypatch,
):
    original = pilot.run_fresh_task_probe
    reports, work, calls = [], [], []
    for selected in (None, False, True):
        fake_environment(monkeypatch)
        options_seen = []

        def run_probe(*args, **options):
            options_seen.append(options.copy())
            return original(*args, **options)

        monkeypatch.setattr(pilot, "run_fresh_task_probe", run_probe)
        report = (
            pilot.run_learnability_pilot()
            if selected is None
            else pilot.run_learnability_pilot(decision_diagnostics=selected)
        )
        assert report["status"] == "complete", report["runs"]
        reports.append(report)
        calls.append([set(options) for options in options_seen])
        assert len(options_seen) == 4
        assert all(
            (options.get("decision_diagnostics") is True)
            if selected is True
            else "decision_diagnostics" not in options
            for options in options_seen
        )
        work.append(
            (
                len(Learner.trains),
                len(Learner.classifications),
                len(Learner.sessions),
                sum(len(session.inputs) for session in Learner.sessions),
            )
        )
    assert work == [(84, 112, 80, 200)] * 3
    assert calls[0] == calls[1] == [{"budget", "checkpoint_updates"}] * 4
    assert calls[2] == [{"budget", "checkpoint_updates", "decision_diagnostics"}] * 4
    assert normalize(reports[0]) == normalize(reports[1])
    assert normalize(without_decisions(reports[2], reports[0]["protocol"])) == normalize(reports[0])
    for row in reports[2]["runs"]:
        assert row["source_unchanged"] is row["actor_unchanged"] is True
        for cohort in row["probes"]:
            result = cohort["report"]
            assert cohort["charged_updates"] == result["completed_updates"] == 16
            assert result["decision_diagnostics_enabled"] is True
            assert result["reserved_data_bytes"] == 10368
            for point in result["checkpoints"]:
                assert set(point["decision_diagnostics"]) == {"E-red", "E-blue"}
                for context, diagnostic in point["decision_diagnostics"].items():
                    cue = 1 if context == "E-red" else -1
                    assert diagnostic["episode_id"] == f"reserved-E-test-{cue}"
                    assert diagnostic["row_ids"] == [
                        f"reserved-E-test-{cue}:step:{step}" for step in range(8)
                    ]
                    assert diagnostic["sequence_kind"] == "complete_episode"
                    assert diagnostic["class_counts"] == [1, 0, 1, 4, 2]
                    assert diagnostic["predicted_class_counts"] == [8, 0, 0, 0, 0]
                    assert diagnostic["per_class"][1]["accuracy"] is None
                    assert diagnostic["per_class"][1]["mean_loss"] is None
                    assert sum(row["correct"] for row in diagnostic["decisions"]) == 1
                    assert point["scores"][context]["accuracy"] == 1 / 8


@pytest.mark.parametrize("endpoint", ["describe_learnability_protocol", "run_learnability_pilot"])
@pytest.mark.parametrize("flag", [0, 1, None, "true", np.bool_(True)])
def test_should_refuse_non_bool_python_flags_before_resources_or_data(monkeypatch, endpoint, flag):
    for name in (
        "ProcessRssSampler",
        "require_torch",
        "build_stream",
        "build_probe_stream",
        "run_fresh_task_probe",
        "KeyDoorNavigation",
    ):
        monkeypatch.setattr(pilot, name, forbidden)
    with pytest.raises(ValueError):
        getattr(pilot, endpoint)(decision_diagnostics=flag)


@pytest.mark.parametrize("missing", ["enabled", "context"])
def test_should_preserve_full_probe_charges_when_requested_diagnostics_are_missing(
    monkeypatch, missing
):
    fake_environment(monkeypatch)
    original = pilot.run_fresh_task_probe

    def incomplete(*args, **options):
        result = original(*args, **options)
        assert options["decision_diagnostics"] is True
        if missing == "enabled":
            result.pop("decision_diagnostics_enabled")
        else:
            result["checkpoints"][0]["decision_diagnostics"].pop("E-blue")
        return result

    monkeypatch.setattr(pilot, "run_fresh_task_probe", incomplete)
    report = pilot.run_learnability_pilot(decision_diagnostics=True)
    assert report["status"] == "incomplete"
    for row in report["runs"]:
        assert (
            row["status"] == "failed" and row["wake_updates"] == row["replay_charged_examples"] == 0
        )
        first = row["probes"][0]
        assert first["status"] == "failed" and first["charged_updates"] == 16
        assert first["report"]["status"] == "complete"
        assert "diagnostic" in first["reason"]
        assert row["probes"][1]["status"] == "not_run"
        assert all(episode["status"] == "not_run" for episode in row["episodes"])


def test_should_keep_real_derived_helper_failure_costs_and_stop_dependent_benchmark_work(
    monkeypatch,
):
    fake_environment(monkeypatch)
    original = probe_module.decision_class_diagnostics
    calls = 0

    def fail(predictions, targets, num_classes):
        nonlocal calls
        calls += 1
        # Each arm's first probe gets two initial contexts, then fails at the
        # first context after four returned training episodes.
        if calls in (3, 6):
            raise RuntimeError("reserved class detail failed")
        return original(predictions, targets, num_classes)

    monkeypatch.setattr(probe_module, "decision_class_diagnostics", fail)
    report = pilot.run_learnability_pilot(decision_diagnostics=True)
    assert report["status"] == "incomplete" and calls == 6
    for row in report["runs"]:
        first = row["probes"][0]
        assert row["wake_updates"] == row["replay_charged_examples"] == 0
        assert first["charged_updates"] == first["report"]["completed_updates"] == 4
        assert (
            first["report"]["score_forward_calls_attempted"]
            == first["report"]["score_forward_calls_completed"]
            == 3
        )
        assert "reserved class detail failed" in first["report"]["reason"]
        assert [point["updates"] for point in first["report"]["checkpoints"]] == [0]
        assert row["probes"][1]["status"] == "not_run"
        assert all(episode["status"] == "not_run" for episode in row["episodes"])


@pytest.mark.parametrize("describe", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_should_keep_zero_argument_cli_callbacks_and_pass_true_only_when_requested(
    monkeypatch, capsys, describe, enabled
):
    selected = "describe_learnability_protocol" if describe else "run_learnability_pilot"
    argv = (
        ["pilot", "--scenario", "torch-key-door-learnability"]
        + (["--describe"] if describe else [])
        + (["--probe-decision-diagnostics"] if enabled else [])
    )
    monkeypatch.setattr(sys, "argv", argv)

    def ordinary():
        return dict(status="complete", requested=False)

    def requested(*, decision_diagnostics):
        assert decision_diagnostics is True
        return dict(status="complete", requested=True)

    for name in ("describe_learnability_protocol", "run_learnability_pilot"):
        monkeypatch.setattr(
            cli, name, (requested if enabled else ordinary) if name == selected else forbidden
        )
    cli.main()
    assert json.loads(capsys.readouterr().out)["requested"] is enabled


@pytest.mark.parametrize(
    "scenario", [None, "torch-key-door-recurrent", "torch-key-door-warm-start-control"]
)
def test_should_reject_diagnostic_flag_for_other_routes_before_any_selected_callback(
    monkeypatch, scenario
):
    for name, value in list(vars(cli).items()):
        if name.startswith(("describe_", "run_")) and callable(value):
            monkeypatch.setattr(cli, name, forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        ["pilot", "--probe-decision-diagnostics"]
        + ([] if scenario is None else ["--scenario", scenario]),
    )
    with pytest.raises(SystemExit) as failure:
        cli.main()
    assert failure.value.code == 2
