"""Bounded saved FIFO JSON IO and resource-free CLI preview, without runtime."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

import scripts.run_minigrid_saved_fifo_readouts as cli

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "artifacts/runs/minigrid-recurring-fifo-20261017/native.json"
)


def forbidden(*args, **kwargs):
    pytest.fail("pure CLI boundary reached source loading or native evaluation")


@pytest.mark.parametrize(
    "raw",
    [
        b'{"x":NaN}',
        b'{"x":Infinity}',
        b'{"x":1e999}',
        b'{"x":1,"x":2}',
        b"\xff",
        b'{"broken":',
        b"[]",
    ],
)
def test_should_refuse_invalid_nonfinite_duplicate_or_nonobject_json(tmp_path, raw):
    path = tmp_path / "bad.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        cli.load_inputs(path)
    assert path.read_bytes() == raw


def test_should_bound_source_before_decoding(tmp_path, monkeypatch):
    path = tmp_path / "large.json"
    path.write_bytes(b" " * 65)
    monkeypatch.setattr(cli, "MAX_INPUT_BYTES", 64)
    with pytest.raises(ValueError):
        cli.load_inputs(path)


def test_should_describe_without_loading_saved_source_or_running(monkeypatch, capsys):
    expected = {"protocol_id": "preview-only", "new_training_updates": 0}
    monkeypatch.setattr(cli, "describe_protocol", lambda: deepcopy(expected))
    monkeypatch.setattr(cli, "load_inputs", forbidden)
    monkeypatch.setattr(cli, "run_pilot", forbidden)
    assert cli.main(["--describe"]) == 0
    assert json.loads(capsys.readouterr().out) == expected


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--source", "x.json"],
        ["--describe", "--source", "x.json"],
        ["--describe", "--output", "x.json"],
    ],
)
def test_should_refuse_missing_source_or_mixed_preview_before_loading(monkeypatch, arguments):
    monkeypatch.setattr(cli, "load_inputs", forbidden)
    monkeypatch.setattr(cli, "run_pilot", forbidden)
    with pytest.raises(SystemExit) as caught:
        cli.main(arguments)
    assert caught.value.code == 2


@pytest.mark.parametrize("seconds", ["0", "-1", "121", "nan", "inf"])
def test_should_refuse_invalid_or_increased_body_allowance_before_loading(monkeypatch, seconds):
    monkeypatch.setattr(cli, "load_inputs", forbidden)
    monkeypatch.setattr(cli, "run_pilot", forbidden)
    with pytest.raises((SystemExit, ValueError)):
        cli.main(["--source", str(SOURCE), "--max-seconds", seconds])


@pytest.mark.parametrize("status,exit_code", [("complete", 0), ("incomplete", 2)])
def test_should_publish_one_complete_or_negative_attempt_and_preserve_origin(
    tmp_path, monkeypatch, capsys, status, exit_code
):
    path = tmp_path / "readouts.json"
    before = SOURCE.read_bytes()
    inputs = object()
    seen = []
    monkeypatch.setattr(cli, "load_inputs", lambda source: inputs)

    def run(received, *, max_seconds):
        seen.append((received, max_seconds))
        return {
            "status": status,
            "reason": "original spent failure preserved",
            "origin": {"status": "incomplete", "training_updates": 128},
            "new_training_updates": 0,
            "censored_games": [[64, "B", "blue"]],
        }

    monkeypatch.setattr(cli, "run_pilot", run)
    assert (
        cli.main(["--source", str(SOURCE), "--output", str(path), "--max-seconds", "10"])
        == exit_code
    )
    assert seen == [(inputs, 10.0)]
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["status"] == status
    assert report["origin"] == {"status": "incomplete", "training_updates": 128}
    assert report["new_training_updates"] == 0
    assert report["censored_games"] == [[64, "B", "blue"]]
    assert json.loads(capsys.readouterr().out) == {
        "status": status,
        "output": str(path),
        "output_bytes": len(path.read_bytes()),
    }
    assert SOURCE.read_bytes() == before


def test_should_refuse_existing_output_before_source_or_runtime(tmp_path, monkeypatch):
    path = tmp_path / "existing.json"
    path.write_bytes(b"unrelated user data")
    monkeypatch.setattr(cli, "load_inputs", forbidden)
    monkeypatch.setattr(cli, "run_pilot", forbidden)
    with pytest.raises(SystemExit) as caught:
        cli.main(["--source", str(SOURCE), "--output", str(path)])
    assert caught.value.code == 2 and path.read_bytes() == b"unrelated user data"


def test_should_refuse_source_output_collision_before_runtime(monkeypatch):
    before = SOURCE.read_bytes()
    monkeypatch.setattr(cli, "load_inputs", forbidden)
    monkeypatch.setattr(cli, "run_pilot", forbidden)
    with pytest.raises(SystemExit):
        cli.main(["--source", str(SOURCE), "--output", str(SOURCE)])
    assert SOURCE.read_bytes() == before


def test_should_preserve_racing_output_instead_of_overwriting(tmp_path, monkeypatch):
    path = tmp_path / "raced.json"
    monkeypatch.setattr(cli, "load_inputs", lambda source: object())

    def run(inputs, *, max_seconds):
        path.write_bytes(b"created during evaluation")
        return {"status": "incomplete", "reason": "paid failure"}

    monkeypatch.setattr(cli, "run_pilot", run)
    with pytest.raises(FileExistsError):
        cli.main(["--source", str(SOURCE), "--output", str(path)])
    assert path.read_bytes() == b"created during evaluation"


@pytest.mark.parametrize("payload", [{"text": "x" * 1000}, {"value": float("inf")}])
def test_should_refuse_oversized_or_nonfinite_report_before_publication(
    tmp_path, monkeypatch, payload
):
    path = tmp_path / "bad-output.json"
    monkeypatch.setattr(cli, "load_inputs", lambda source: object())
    monkeypatch.setattr(
        cli, "run_pilot", lambda inputs, **options: {"status": "incomplete", **payload}
    )
    monkeypatch.setattr(cli, "MAX_OUTPUT_BYTES", 64)
    with pytest.raises(ValueError):
        cli.main(["--source", str(SOURCE), "--output", str(path)])
    assert not path.exists()
