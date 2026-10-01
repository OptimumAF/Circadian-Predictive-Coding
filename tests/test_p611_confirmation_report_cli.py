"""Dry CLI and exact port composition; all score/data/model work is sealed."""

from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from scripts import run_p611_confirmation_report as adapter


@pytest.mark.parametrize(
    "arguments,code",
    [
        (["--help"], 0),
        ([], 2),
        (["--publish", "--read-only"], 2),
        (["--publish", "--seed", "41"], 2),
        (["--publish", "--alpha", "0.1"], 2),
    ],
)
def test_should_be_dry_or_reject_scientific_overrides(arguments: list[str], code: int) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.run_p611_confirmation_report", *arguments],
        cwd=adapter.REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert completed.returncode == code


def test_should_supply_the_unchanged_complete_cost_reader_and_canonical_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def read(path: Path, *, read_only: bool) -> dict[str, Any]:
        calls.append((path, read_only))
        return {"fixture_only": True}

    monkeypatch.setattr(adapter, "inspect_confirmation_costs", read)
    assert adapter._costs() == {"fixture_only": True}
    assert calls == [(adapter.REPO_ROOT / adapter.COST_FILES[0], True)]


def test_should_supply_the_unchanged_complete_scored_reader_and_fresh_reference_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callback = lambda: {"fixture_only": True}
    directory = adapter.REPO_ROOT / "fixture"

    def read(root: Path, path: Path, scope: Path, references: Any) -> Any:
        assert root == adapter.REPO_ROOT and path == directory and scope == adapter.SCOPE_FILE
        assert references is callback
        return ({}, {}, {})

    monkeypatch.setattr(adapter, "read_completed_scored_bundle", read)
    assert adapter._scored(directory, callback) == ({}, {}, {})
