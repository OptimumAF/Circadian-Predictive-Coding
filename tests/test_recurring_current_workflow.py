"""Current-only ordinary workflow/CLI, real stream and literal numerical ports."""

from copy import deepcopy
from dataclasses import replace
import json
import sys

import pytest

from scripts import minigrid_affine_workflow as workflow
from scripts import run_minigrid_linear_fit_control as cli
from test_recurring_current_control import bind_ports, sources


@pytest.fixture
def scripted(monkeypatch):
    box = bind_ports(monkeypatch)
    packets = tuple(
        replace(p, context=context)
        for p, context in zip(
            sources(), ("A", "A", "B", "B", "C_train", "C_train", "G_train", "G_train"), strict=True
        )
    )
    box.update(collection=[], games=[], training=[])
    monkeypatch.setattr(workflow, "_require_minigrid", lambda: None)
    monkeypatch.setattr(cli, "read_training_sources", lambda *a: packets[:6])
    original = cli.run_context_linear_stream

    def train(*args, **options):
        assert options["recurring_catalogue"] is True
        assert options["retain_arrived"] is False and options["recurring_current_only"] is True
        report = original(*args, **options)
        box["training"].append(deepcopy(report))
        box["live_training"] = report
        if box.get("effect"):
            box["effect"](report)
        return report

    def collect(check, reports):
        for p in packets[-2:]:
            reports.append(
                dict(
                    status="complete",
                    closed=True,
                    role="train",
                    context="G_train",
                    episode_id=p.episode_id,
                    scripted=True,
                )
            )
            check()
        box["collection"] = reports
        return packets[-2:]

    def games(source, state, **options):
        assert len(box["fits"]) == 10 and len(box["scores"]) == 48
        assert source.snapshot_state().completed_updates == 0
        assert state.completed_updates == 10 and state.training_examples == 252
        assert options["recurring_catalogue"] is True
        assert (
            options["budget"].updates_completed == options["budget"].replay_examples_completed == 0
        )
        options["check_boundary"]()
        result = dict(
            status="complete",
            state_unchanged=True,
            episodes=[
                dict(status="complete", success=False, environment=dict(closed=True))
                for _ in range(4)
            ],
        )
        box["games"].append(result)
        return result

    monkeypatch.setattr(cli, "run_context_linear_stream", train)
    monkeypatch.setattr(workflow, "_collect_recurring_train_sources", collect)
    monkeypatch.setattr(workflow, "evaluate_context_linear_stream_checkpoint", games)
    return box


def run(**options):
    return workflow.run_workflow(
        **{"recurring_training": True, "recurring_current_only": True, **options}
    )


def test_should_execute_actual_delayed_current_repeat_stream_before_known_game_dispatch(scripted):
    result = run()
    assert result["status"] == "complete", result["reason"]
    assert result["definition"] == "portable_affine_recurring_current_only_v1"
    assert result["attempted_phases"] == ["collection", "training", "games"]
    assert len(result["collection"]) == 2 and all(p["closed"] for p in result["collection"])
    assert len(result["games"]["episodes"]) == 4 and not any(
        e["success"] for e in result["games"]["episodes"]
    )
    assert result["training"]["current_repetition"]["committed_rows"] == dict(
        arrivals=90, repeated_current=162, native=252
    )
    assert "catalogue" not in result["training"]
    assert result["readout_budget"]["initial_limits"] == result["readout_budget"]["limits"]
    assert (
        result["readout_budget"]["started_at"]
        == result["readout_budget"]["observed_budget_started_at"]
    )


@pytest.mark.parametrize(
    "options",
    [
        dict(recurring_current_only=1),
        dict(recurring_current_only=None),
        dict(recurring_training=False),
        dict(private_adaptation=True),
        dict(a4_adaptation=True),
        dict(train_update="minimum_change"),
        dict(probe_update="minimum_change"),
        dict(all_old_tasks=True),
        dict(joint_heldout=True),
    ],
)
def test_should_refuse_ambiguous_or_private_intents_before_data(monkeypatch, options):
    monkeypatch.setattr(workflow, "example_affine_training", lambda: pytest.fail("data reached"))
    arguments = {"recurring_training": True, "recurring_current_only": True, **options}
    for action in (workflow.describe_workflow, workflow.run_workflow):
        with pytest.raises(ValueError):
            action(**arguments)


def test_should_leave_original_joint_description_identical_when_option_omitted_or_false():
    assert workflow.describe_workflow(recurring_training=True) == workflow.describe_workflow(
        recurring_training=True, recurring_current_only=False
    )
    current = workflow.describe_workflow(recurring_training=True, recurring_current_only=True)
    assert current["training"] == cli.describe_context_linear_recurring_current_only()
    assert current["new_fit_calls"] == 10 and current["new_fit_rows"] == 252


@pytest.mark.parametrize(
    "fault",
    ["definition", "protocol", "prior_rows", "bank", "repeats", "unknown", "receipt", "joint"],
)
def test_should_refuse_wrong_control_evidence_even_when_endpoint_shape_matches(scripted, fault):
    def corrupt(report):
        cost = report["current_repetition"]
        if fault == "definition":
            report["definition"] = cli.describe_context_linear_recurring()["definition"]
        elif fault == "protocol":
            report["protocol"]["current_packet_multiplicities"] = (1,) * 10
        elif fault == "prior_rows":
            cost["prior_packet_replay_rows"] = 162
        elif fault == "bank":
            cost["retained_bank_rows"] = 54
        elif fault == "repeats":
            cost["repeated_current_row_exposures"] = 0
        elif fault == "unknown":
            cost["failure_work_may_be_unobserved"] = True
        elif fault == "receipt":
            cost["verified_receipts"][0]["sample_id"] = "replaced"
        else:
            report["catalogue"] = {}

    scripted["effect"] = corrupt
    result = run()
    assert result["status"] == "failed" and "current-repeat evidence" in result["reason"]
    assert result["games"] is None and not scripted["games"]
    assert len(result["training"]["committed_receipts"]) == 10


def test_should_retain_lost_ninth_fit_receipts_without_admitting_games(scripted):
    scripted["fail_at"] = 9
    result = run()
    assert result["status"] == "failed" and result["games"] is None
    training = result["training"]
    assert len(training["committed_receipts"]) == training["budget"]["updates_completed"] == 8
    assert training["current_repetition"]["committed_rows"]["native"] == 180
    assert training["failure_work"]["work"]["fit_rows_attempted"] == 36
    assert not scripted["scores"] and not scripted["games"]


@pytest.mark.parametrize("fault", ["collection", "training", "clock"])
def test_should_keep_original_paid_records_if_readout_boundary_changes(
    scripted, monkeypatch, fault
):
    def games(*args, budget, check_boundary, **options):
        if fault == "collection":
            scripted["collection"][0] = dict(scripted["collection"][0])
        elif fault == "training":
            scripted["live_training"]["budget"]["updates_completed"] = 11
        else:
            budget.started_at += 1
        check_boundary()
        pytest.fail("changed readout evidence admitted")

    monkeypatch.setattr(workflow, "evaluate_context_linear_stream_checkpoint", games)
    result = run()
    assert result["status"] == "failed"
    assert result["training"] == scripted["training"][0]
    assert len(result["collection"]) == 2 and all(x["closed"] for x in result["collection"])


def test_should_describe_cli_without_data_native_or_execution(monkeypatch, capsys):
    monkeypatch.setattr(workflow, "run_workflow", lambda **k: pytest.fail("execute reached"))
    monkeypatch.setattr(workflow, "_require_minigrid", lambda: pytest.fail("native reached"))
    monkeypatch.setattr(
        sys,
        "argv",
        ["control", "--example-affine-recurring", "--recurring-current-only", "--describe"],
    )
    cli.main()
    result = json.loads(capsys.readouterr().out)
    assert result["definition"] == "portable_affine_recurring_current_only_v1"
    assert result["training"]["repeated_current_row_exposures"] == 162


def test_should_dispatch_cli_once_to_explicit_control(monkeypatch, capsys):
    calls = []

    def action(**options):
        calls.append(options)
        return dict(status="complete")

    monkeypatch.setattr(workflow, "run_workflow", action)
    monkeypatch.setattr(
        sys, "argv", ["control", "--example-affine-recurring", "--recurring-current-only"]
    )
    cli.main()
    capsys.readouterr()
    assert calls == [dict(recurring_training=True, recurring_current_only=True)]


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--example-affine-play"],
        ["--example-affine-adaptation"],
        ["--example-affine-recurring", "--train-update", "minimum_change"],
    ],
)
def test_should_refuse_unrelated_cli_modes_before_dispatch(monkeypatch, arguments):
    monkeypatch.setattr(workflow, "run_workflow", lambda **k: pytest.fail("execute reached"))
    monkeypatch.setattr(sys, "argv", ["control", "--recurring-current-only", *arguments])
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2
