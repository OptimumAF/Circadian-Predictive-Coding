"""The fixed CLI composes the unchanged original training reader explicitly."""

from pathlib import Path
import sys
from typing import Any

import pytest

from scripts import run_p610_retention_costs as adapter


@pytest.mark.parametrize("mode", ["--publish", "--read-only"])
def test_should_dispatch_fixed_modes_with_the_actual_training_reader(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = []

    def original(directory: Path, scope_file: Path) -> Any:
        calls.append((directory, scope_file))
        return ({}, {}, {})

    def operation(root: Path, directory: Path, scope: Path, reader: Any) -> Any:
        assert root == adapter.REPO_ROOT and scope == adapter.SCOPE_FILE and directory == tmp_path
        assert reader(tmp_path) == ({}, {}, {})
        return (
            ({}, {}, {"status": "fixture_completed"})
            if mode == "--read-only"
            else {"status": "fixture_completed"}
        )

    monkeypatch.setattr(adapter.training_adapter, "read_completed_bundle", original)
    monkeypatch.setattr(
        adapter,
        "publish_retention_costs" if mode == "--publish" else "read_completed_retention_costs",
        operation,
    )
    monkeypatch.setattr(
        sys, "argv", ["run_p610_retention_costs", mode, "--output-dir", str(tmp_path)]
    )
    adapter.main()
    assert calls == [(tmp_path, adapter.SCOPE_FILE)]
    assert "fixture_completed" in capsys.readouterr().out


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--publish", "--read-only"],
        ["--publish", "--seeds", "1"],
        ["--publish", "--validation-seconds", "999"],
        ["--publish", "--scope-file", "different"],
    ],
)
def test_should_refuse_missing_conflicting_or_scientific_overrides(
    monkeypatch: pytest.MonkeyPatch, arguments: list[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["run_p610_retention_costs", *arguments])
    with pytest.raises(SystemExit) as result:
        adapter.main()
    assert result.value.code == 2
