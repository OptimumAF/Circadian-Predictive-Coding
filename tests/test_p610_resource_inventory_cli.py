"""Fixed resource CLI dispatch, with no scientific authority from these spies."""

import json
from pathlib import Path
import sys
from typing import Any

import pytest

from scripts import run_p610_resource_inventory as cli


@pytest.mark.parametrize("mode", ["--publish", "--read-only"])
def test_should_dispatch_fixed_inventory_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], mode: str
) -> None:
    calls = []
    audit = {"status": "completed", "coverage": {"fixture_only": True}}

    def publish(*args: Any) -> Any:
        calls.append(args)
        return audit

    def read(*args: Any) -> Any:
        calls.append(args)
        return {}, {}, audit

    monkeypatch.setattr(cli, "publish_resource_inventory", publish)
    monkeypatch.setattr(cli, "read_completed_resource_inventory", read)
    directory = tmp_path / "inventory"
    monkeypatch.setattr(sys, "argv", ["inventory", mode, "--output-dir", str(directory)])
    cli.main()
    assert calls == [(cli.ROOT, directory, cli.SCOPE_FILE)]
    assert json.loads(capsys.readouterr().out) == audit


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--publish", "--read-only"],
        ["--publish", "--seed", "1"],
        ["--publish", "--metric", "favorable"],
        ["--publish", "--budget-seconds", "999"],
        ["--publish", "--source", "unbound"],
    ],
)
def test_should_reject_conflicts_and_overrides_before_reading_inputs(
    monkeypatch: pytest.MonkeyPatch, arguments: list[str]
) -> None:
    def forbid(*args: Any) -> Any:
        raise AssertionError("invalid CLI entered inventory boundary")

    monkeypatch.setattr(cli, "publish_resource_inventory", forbid)
    monkeypatch.setattr(cli, "read_completed_resource_inventory", forbid)
    monkeypatch.setattr(sys, "argv", ["inventory", *arguments])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_should_show_help_without_reading_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbid(*args: Any) -> Any:
        raise AssertionError("help entered inventory boundary")

    monkeypatch.setattr(cli, "publish_resource_inventory", forbid)
    monkeypatch.setattr(cli, "read_completed_resource_inventory", forbid)
    monkeypatch.setattr(sys, "argv", ["inventory", "--help"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 0
