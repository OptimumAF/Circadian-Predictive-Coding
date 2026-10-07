"""The adapter exposes only stored evidence projection and exclusive output."""

from pathlib import Path
import sys
from typing import Any

import pytest

from scripts import inspect_p612_development_findings as cli


def test_should_refuse_occupied_output_before_any_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "occupied.json"
    path.write_bytes(b"preserve original user output")

    def forbidden() -> Any:
        pytest.fail("occupied output invoked validation")

    monkeypatch.setattr(cli, "inspect_development_findings", forbidden)
    with pytest.raises(FileExistsError):
        cli.publish_development_findings(path)
    assert path.read_bytes() == b"preserve original user output"


def test_should_compose_all_ports_and_propagate_complete_inputs_without_a_scientific_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}
    inputs, ledger = {"fixture_only": "whole inputs"}, {"fixture_only": "ledger"}

    def reader(root: Any, verifiers: Any, preflight: Any) -> Any:
        observed["ports"] = tuple(verifiers)
        assert root == cli.original.REPO_ROOT and preflight is cli._preflight
        return inputs

    def builder(value: Any) -> Any:
        assert value is inputs
        return ledger

    monkeypatch.setattr(cli, "read_development_inputs", reader)
    monkeypatch.setattr(cli, "build_development_ledger", builder)
    monkeypatch.setattr(
        cli,
        "verify_development_input_bindings",
        lambda root, value: observed.update(late=value is inputs),
    )
    assert cli.inspect_development_findings() is ledger
    assert observed["ports"] == ("gating", "replay", "sleep", "schedule", "combined", "parent")
    assert observed["late"] is True


@pytest.mark.parametrize("flag", ["--worker", "--seed", "--metric", "--train", "--final"])
def test_should_expose_no_scientific_override_or_worker(
    flag: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        sys, "argv", ["inspect_p612_development_findings", "--output-file", "unused.json", flag]
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
