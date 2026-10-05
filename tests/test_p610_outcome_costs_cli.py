"""The fixed presentation CLI composes the whole unchanged report reader."""

from pathlib import Path
import sys
from typing import Any

import pytest

from scripts import run_p610_outcome_costs as adapter


@pytest.mark.parametrize("mode", ["--publish", "--read-only"])
def test_should_dispatch_fixed_modes_through_the_complete_report_reader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = []

    def original(*args: Any) -> Any:
        calls.append(args)
        return ({}, {}, {})

    def operation(root: Path, directory: Path, scope: Path, reader: Any) -> Any:
        assert root == adapter.REPO_ROOT and scope == adapter.SCOPE_FILE and directory == tmp_path
        assert reader() == ({}, {}, {})
        audit = {"status": "fixture_completed"}
        return ({}, {}, audit) if mode == "--read-only" else audit

    monkeypatch.setattr(adapter, "read_completed_confirmation_report", original)
    monkeypatch.setattr(
        adapter,
        "publish_outcome_costs" if mode == "--publish" else "read_completed_outcome_costs",
        operation,
    )
    monkeypatch.setattr(
        sys, "argv", ["run_p610_outcome_costs", mode, "--output-dir", str(tmp_path)]
    )
    adapter.main()
    assert calls == [
        (
            adapter.REPO_ROOT,
            adapter.REPO_ROOT / "artifacts/runs/p611-confirmation-report",
            adapter.SCOPE_FILE,
            adapter._scored,
            adapter._costs,
        )
    ]
    assert "fixture_completed" in capsys.readouterr().out


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--publish", "--read-only"],
        ["--publish", "--seeds", "1"],
        ["--publish", "--validation-seconds", "999"],
        ["--publish", "--scope-file", "different"],
        ["--publish", "--metric", "new"],
    ],
)
def test_should_refuse_missing_conflicting_or_scientific_overrides(
    monkeypatch: pytest.MonkeyPatch,
    arguments: list[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["run_p610_outcome_costs", *arguments])
    with pytest.raises(SystemExit) as result:
        adapter.main()
    assert result.value.code == 2
