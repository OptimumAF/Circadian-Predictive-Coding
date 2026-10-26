"""Saved-data CLI uses the pure comparison boundary; no learners or collection."""

import json
import sys

import pytest

from scripts import compare_minigrid_rehearsal as cli
from test_minigrid_rehearsal_comparison import pair


def test_should_dispatch_saved_reports_without_model_work(monkeypatch, tmp_path, capsys):
    paths = [tmp_path / "control.json", tmp_path / "replay.json"]
    records = [dict(control=True), dict(replay=True)]
    for path, record in zip(paths, records, strict=True):
        path.write_text(json.dumps(record), encoding="utf8")
    calls = []

    def compare(control, replay, *, method):
        calls.append((control, replay, method))
        return dict(status="qualified")

    monkeypatch.setattr(cli, "compare_reports", compare)
    monkeypatch.setattr(
        sys,
        "argv",
        ["compare", "--control", str(paths[0]), "--rehearsal", str(paths[1]), "--method", "bp"],
    )
    cli.main()
    assert calls == [(*records, "bp")]
    assert json.loads(capsys.readouterr().out) == dict(status="qualified")


def test_should_compare_complete_saved_pair_through_public_cli(monkeypatch, tmp_path, capsys):
    # Replay DTO is deliberately fabricated in pair(); this tests saved arithmetic,
    # never claims the synthetic replay branch is an observed scientific result.
    paths = [tmp_path / "control.json", tmp_path / "replay.json"]
    for path, report in zip(paths, pair(), strict=True):
        path.write_text(json.dumps(report), encoding="utf8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["compare", "--control", str(paths[0]), "--rehearsal", str(paths[1]), "--method", "bp"],
    )
    cli.main()
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "complete"
    assert result["extra_native_calls"] == 64 and result["extra_training_rows"] == 576
    assert result["scientific_advantage_verified"] is False


@pytest.mark.parametrize("problem", ["missing", "invalid_json", "incomplete"])
def test_should_explain_refused_saved_comparison(monkeypatch, tmp_path, capsys, problem):
    path = tmp_path / "report.json"
    if problem != "missing":
        path.write_text("not JSON" if problem == "invalid_json" else "{}", encoding="utf8")

    def refuse(*args, **kwargs):
        raise ValueError("incomplete input")

    monkeypatch.setattr(cli, "compare_reports", refuse)
    monkeypatch.setattr(
        sys, "argv", ["compare", "--control", str(path), "--rehearsal", str(path), "--method", "bp"]
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert "saved MiniGrid comparison refused" in capsys.readouterr().err
