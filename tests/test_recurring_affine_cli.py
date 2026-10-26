"""CLI delivery boundaries for the existing recurring recipe; zero learning."""

import json
import sys

import pytest

from scripts import minigrid_affine_workflow as workflow
from scripts import run_minigrid_linear_fit_control as cli

RECURRING = "--example-affine-recurring"


@pytest.mark.parametrize(
    "flag,options",
    [
        ("--example-affine-play", {}),
        ("--example-affine-adaptation", {"private_adaptation": True}),
        (RECURRING, {"recurring_training": True}),
    ],
)
def test_should_describe_each_original_or_recurring_recipe_before_work(
    monkeypatch, capsys, flag, options
):
    expected = json.loads(json.dumps(workflow.describe_workflow(**options), allow_nan=False))
    monkeypatch.setattr(workflow, "run_workflow", lambda **k: pytest.fail("execution reached"))
    monkeypatch.setattr(workflow, "_require_minigrid", lambda: pytest.fail("runtime reached"))
    monkeypatch.setattr(
        workflow, "example_affine_training", lambda: pytest.fail("resource reached")
    )
    monkeypatch.setattr(cli, "read_training_sources", lambda *a: pytest.fail("source reached"))
    monkeypatch.setattr(sys, "argv", ["control", flag, "--describe"])
    cli.main()
    assert json.loads(capsys.readouterr().out) == expected


@pytest.mark.parametrize(
    "arguments",
    [
        ["--example-affine-play"],
        ["--example-affine-adaptation"],
        ["--context-linear-fit"],
        ["--source", "missing"],
        ["--retained-source", "missing"],
        ["--fit-updates", "2"],
        ["--fit-seconds", "1"],
        ["--cue-turn-gradients"],
    ],
)
def test_should_refuse_conflicting_recurring_arguments_before_execution(monkeypatch, arguments):
    monkeypatch.setattr(workflow, "run_workflow", lambda **k: pytest.fail("execution reached"))
    monkeypatch.setattr(sys, "argv", ["control", RECURRING, *arguments])
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2


@pytest.mark.parametrize("argument", ["--source", "--output"])
def test_should_refuse_describe_with_io_arguments(monkeypatch, argument):
    monkeypatch.setattr(workflow, "run_workflow", lambda **k: pytest.fail("execution reached"))
    monkeypatch.setattr(sys, "argv", ["control", RECURRING, "--describe", argument, "missing"])
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2


@pytest.mark.parametrize(
    "flag,expected",
    [
        ("--example-affine-play", {}),
        ("--example-affine-adaptation", {"private_adaptation": True}),
        (RECURRING, {"recurring_training": True}),
    ],
)
def test_should_dispatch_only_the_selected_existing_recipe(monkeypatch, capsys, flag, expected):
    calls = []

    def run(**options):
        calls.append(options)
        return dict(status="complete")

    monkeypatch.setattr(workflow, "run_workflow", run)
    monkeypatch.setattr(cli, "read_training_sources", lambda *a: pytest.fail("source reached"))
    monkeypatch.setattr(sys, "argv", ["control", flag])
    cli.main()
    assert calls == [expected] and json.loads(capsys.readouterr().out)["status"] == "complete"


def test_should_create_exclusive_output_and_refuse_overwrite_before_more_work(
    monkeypatch, tmp_path, capsys
):
    output = tmp_path / "report.json"
    calls = []

    def run(**options):
        calls.append(options)
        return dict(status="complete", training=dict(prefix=10))

    monkeypatch.setattr(workflow, "run_workflow", run)
    monkeypatch.setattr(sys, "argv", ["control", RECURRING, "--output", str(output)])
    cli.main()
    raw = output.read_bytes()
    assert json.loads(raw) == json.loads(capsys.readouterr().out)
    with pytest.raises(FileExistsError):
        cli.main()
    assert output.read_bytes() == raw and calls == [dict(recurring_training=True)]


def test_should_save_structured_failure_before_returning_exit_two(monkeypatch, tmp_path, capsys):
    output = tmp_path / "failed.json"
    failed = dict(
        status="failed", reason="scripted paid prefix", training=dict(completed_updates=2)
    )
    monkeypatch.setattr(workflow, "run_workflow", lambda **k: failed)
    monkeypatch.setattr(sys, "argv", ["control", RECURRING, "--output", str(output)])
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2
    assert json.loads(output.read_bytes()) == json.loads(capsys.readouterr().out)
    assert json.loads(output.read_bytes())["training"]["completed_updates"] == 2


def test_should_refuse_missing_runtime_in_actual_workflow_before_data_or_models(
    monkeypatch, capsys
):
    def missing():
        raise RuntimeError("missing pinned optional game runtime")

    monkeypatch.setattr(workflow, "_require_minigrid", missing)
    monkeypatch.setattr(
        workflow, "example_affine_training", lambda: pytest.fail("resource reached")
    )
    monkeypatch.setattr(
        workflow, "_collect_recurring_train_sources", lambda *a: pytest.fail("teacher reached")
    )
    monkeypatch.setattr(workflow, "ContextLinearLearner", lambda *a: pytest.fail("model reached"))
    monkeypatch.setattr(cli, "read_training_sources", lambda *a: pytest.fail("source reached"))
    monkeypatch.setattr(sys, "argv", ["control", RECURRING])
    with pytest.raises(SystemExit) as caught:
        cli.main()
    report = json.loads(capsys.readouterr().out)
    assert caught.value.code == 2 and report["status"] == "failed"
    assert "missing pinned optional game runtime" in report["reason"]
    assert report["training"] is report["games"] is None and report["collection"] == []


def test_should_keep_first_party_size_ceiling_before_writing_output(monkeypatch, tmp_path):
    output = tmp_path / "oversized.json"
    monkeypatch.setattr(
        workflow, "run_workflow", lambda **k: dict(status="complete", payload="x" * (4 * 1024**2))
    )
    monkeypatch.setattr(sys, "argv", ["control", RECURRING, "--output", str(output)])
    with pytest.raises(ValueError, match="exceeds4MiB"):
        cli.main()
    assert not output.exists()
