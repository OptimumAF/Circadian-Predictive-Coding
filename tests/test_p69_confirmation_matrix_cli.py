"""Fixed matrix CLI dispatch; metadata spies establish no scientific authority."""

import json
from pathlib import Path
import sys
from typing import Any

import pytest

from scripts import run_p69_confirmation_matrix as cli


def _forbid(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("CLI fixture entered the original complete scientific reader")


@pytest.mark.parametrize("mode", ["--publish", "--read-only"])
def test_should_dispatch_fixed_mode_and_the_original_complete_reader_port(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], mode: str
) -> None:
    calls = []
    audit = {
        "status": "completed",
        "files": {"fixture_only": True},
        "coverage": {"fixture_only": True},
        "report_identity": {"fixture_only": True},
    }

    def publish(*args: Any) -> Any:
        calls.append(args)
        return audit

    def read(*args: Any) -> Any:
        calls.append(args)
        return {}, {}, audit

    monkeypatch.setattr(cli, "_report", _forbid)
    monkeypatch.setattr(cli, "publish_confirmation_matrix", publish)
    monkeypatch.setattr(cli, "read_completed_confirmation_matrix", read)
    directory = tmp_path / "new-matrix"
    monkeypatch.setattr(sys, "argv", ["matrix", mode, "--output-dir", str(directory)])
    cli.main()
    assert calls == [(cli.REPO_ROOT, directory, cli.SCOPE_FILE, _forbid)]
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "completed"
    assert output["new_training_or_final_source_access"] is False
    assert output["original_fully_measured_matrix_acceptance_complete"] is False


def test_should_supply_the_unchanged_complete_report_reader_and_original_ports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def reader(*args: Any) -> Any:
        calls.append(args)
        return {"fixture": "request"}, {"fixture": "result"}, {"fixture": "audit"}

    monkeypatch.setattr(cli, "read_completed_confirmation_report", reader)
    assert len(cli._report()) == 3
    assert calls == [
        (
            cli.REPO_ROOT,
            cli.REPO_ROOT / "artifacts/runs/p611-confirmation-report",
            cli.SCOPE_FILE,
            cli._scored,
            cli._costs,
        )
    ]


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--publish", "--read-only"],
        ["--publish", "--seed", "1"],
        ["--publish", "--metric", "favorable"],
        ["--publish", "--alpha", "1"],
        ["--publish", "--endpoint", "b_after_a"],
        ["--publish", "--budget-seconds", "999"],
    ],
)
def test_should_refuse_missing_mode_conflicts_and_every_scientific_override_before_ports(
    monkeypatch: pytest.MonkeyPatch, arguments: list[str]
) -> None:
    monkeypatch.setattr(cli, "_report", _forbid)
    monkeypatch.setattr(cli, "publish_confirmation_matrix", _forbid)
    monkeypatch.setattr(cli, "read_completed_confirmation_matrix", _forbid)
    monkeypatch.setattr(sys, "argv", ["matrix", *arguments])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_should_show_help_without_any_complete_reader_or_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "_report", _forbid)
    monkeypatch.setattr(cli, "publish_confirmation_matrix", _forbid)
    monkeypatch.setattr(cli, "read_completed_confirmation_matrix", _forbid)
    monkeypatch.setattr(sys, "argv", ["matrix", "--help"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 0
