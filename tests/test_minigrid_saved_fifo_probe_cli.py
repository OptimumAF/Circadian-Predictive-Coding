"""Saved FIFO CLI admission, unchanged old routes and permanent failed outputs."""

from dataclasses import asdict
import json
from pathlib import Path

import pytest

import scripts.run_minigrid_memory_probe as cli


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "artifacts/runs/minigrid-recurring-fifo-20261017/native.json"
)


def forbidden(*args, **kwargs):
    pytest.fail("unexpected loader or runtime execution")


def test_should_load_actual_full_fifo_states_without_retraining_or_rewriting_source():
    before = SOURCE.read_bytes()
    original = json.loads(before)
    inputs = cli.load_saved_fifo_inputs(SOURCE)
    assert inputs.origin_status == "incomplete"
    assert inputs.original_training_updates == 128 and inputs.original_training_rows == 1152
    assert len(inputs.snapshots) == 5 and len(inputs.episodes) == 6
    assert tuple(state.completed_updates for state in inputs.snapshots) == (0, 32, 64, 96, 128)
    for state, observation in zip(inputs.snapshots, original["observations"], strict=True):
        assert json.loads(json.dumps(asdict(state))) == observation["snapshot"]
        assert state.policy.native_policy.max_entries == 36
        assert state.policy.native_policy.max_payload_bytes == 87264
    assert inputs.censored_games == ((64, "B", "blue"),)
    assert SOURCE.read_bytes() == before
    assert cli.MAX_INPUT_BYTES == cli.MAX_OUTPUT_BYTES == 32 * 1024**2
    assert cli.MAX_SAVED_FIFO_INPUT_BYTES == 16 * 1024**2


def test_should_pass_exactly_one_bounded_read_to_existing_decoder(tmp_path, monkeypatch):
    import src.adapters.minigrid_saved_fifo_inputs as decoder

    source = tmp_path / "source.json"
    source.write_bytes(b'{"saved":true}')
    calls = []
    sentinel = object()
    read = cli._read_source

    def counted_read(path, *, max_bytes=None):
        calls.append((path, max_bytes))
        return read(path, max_bytes=max_bytes)

    def decode(payload):
        assert payload == {"saved": True}
        return sentinel

    monkeypatch.setattr(cli, "_read_source", counted_read)
    monkeypatch.setattr(decoder, "decode_saved_fifo_inputs", decode)
    assert cli.load_saved_fifo_inputs(source) is sentinel
    assert calls == [(source, 16 * 1024**2)]


@pytest.mark.parametrize(
    "raw",
    [b'{"x":NaN}', b'{"x":Infinity}', b'{"x":1e999}', b'{"x":1,"x":2}', b"\xff", b"[]"],
)
def test_should_refuse_invalid_saved_json_before_decoder(tmp_path, monkeypatch, raw):
    import src.adapters.minigrid_saved_fifo_inputs as decoder

    source = tmp_path / "invalid.json"
    source.write_bytes(raw)
    monkeypatch.setattr(decoder, "decode_saved_fifo_inputs", forbidden)
    with pytest.raises(ValueError):
        cli.load_saved_fifo_inputs(source)
    assert source.read_bytes() == raw


def test_should_keep_saved_input_limit_distinct_from_old_loader_limit(tmp_path, monkeypatch):
    import src.adapters.minigrid_saved_fifo_inputs as decoder

    source = tmp_path / "large.json"
    source.write_bytes(b'{"padding":"' + b"a" * 64 + b'"}')
    monkeypatch.setattr(cli, "MAX_SAVED_FIFO_INPUT_BYTES", 64)
    monkeypatch.setattr(cli, "MAX_INPUT_BYTES", 128)
    monkeypatch.setattr(decoder, "decode_saved_fifo_inputs", forbidden)
    with pytest.raises(ValueError, match="exceeds"):
        cli.load_saved_fifo_inputs(source)
    assert len(cli._read_source(source)["padding"]) == 64


def test_should_preview_saved_mode_without_any_source_or_runtime(monkeypatch, capsys):
    expected = {"protocol_id": "saved-fifo-return", "benchmark_updates": 0}
    monkeypatch.setattr(cli, "describe_saved_fifo_return_protocol", lambda: expected)
    for name in (
        "load_saved_fifo_inputs",
        "run_saved_fifo_return_pilot",
        "load_snapshots",
        "load_return_inputs",
        "run_pilot",
        "run_return_pilot",
        "describe_protocol",
        "describe_return_protocol",
        "_read_source",
    ):
        monkeypatch.setattr(cli, name, forbidden)
    assert cli.main(["--saved-fifo-return", "--describe"]) == 0
    assert json.loads(capsys.readouterr().out) == expected


@pytest.mark.parametrize("status,code", [("complete", 0), ("incomplete", 2)])
@pytest.mark.parametrize("seconds", [None, "0.125", "120"])
def test_should_route_saved_inputs_once_and_publish_all_outcomes(
    tmp_path, monkeypatch, capsys, status, code, seconds
):
    output = tmp_path / "new-attempt.json"
    calls = []
    sentinel = object()

    def load(path):
        calls.append(("load", path))
        return sentinel

    def run(inputs, *, max_seconds):
        assert inputs is sentinel
        calls.append(("run", max_seconds))
        return {
            "status": status,
            "cohorts": [{"status": "failed", "charged_updates": 3}],
            "reason": "negative or censored result remains recorded",
        }

    monkeypatch.setattr(cli, "load_saved_fifo_inputs", load)
    monkeypatch.setattr(cli, "run_saved_fifo_return_pilot", run)
    for name in ("load_snapshots", "load_return_inputs", "run_pilot", "run_return_pilot"):
        monkeypatch.setattr(cli, name, forbidden)
    args = ["--saved-fifo-return", "--source-artifact", str(SOURCE), "--output", str(output)]
    if seconds is not None:
        args += ["--max-seconds", seconds]
    before = SOURCE.read_bytes()
    assert cli.main(args) == code
    assert calls == [("load", SOURCE), ("run", 120.0 if seconds is None else float(seconds))]
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == status
    assert report["cohorts"][0]["charged_updates"] == 3
    assert report["source_artifact"] == str(SOURCE)
    assert report["source_load_seconds"] >= 0
    assert json.loads(capsys.readouterr().out) == report
    assert SOURCE.read_bytes() == before


@pytest.mark.parametrize(
    "args",
    [
        ["--saved-fifo-return"],
        ["--saved-fifo-return", "--source-artifact", "source.json"],
        ["--saved-fifo-return", "--return-comparison", "--describe"],
        ["--saved-fifo-return", "--describe", "--source-artifact", "source.json"],
        ["--saved-fifo-return", "--describe", "--output", "output.json"],
        ["--max-seconds", "1", "--describe"],
        ["--return-comparison", "--max-seconds", "1", "--describe"],
    ]
    + [
        ["--saved-fifo-return", "--describe", "--max-seconds", value]
        for value in ("0", "-1", "120.001", "nan", "inf", "-inf", "bad")
    ],
)
def test_should_refuse_invalid_arguments_before_loading_or_execution(monkeypatch, args):
    for name in (
        "load_saved_fifo_inputs",
        "run_saved_fifo_return_pilot",
        "load_snapshots",
        "load_return_inputs",
        "run_pilot",
        "run_return_pilot",
        "describe_saved_fifo_return_protocol",
    ):
        monkeypatch.setattr(cli, name, forbidden)
    with pytest.raises(SystemExit) as raised:
        cli.main(args)
    assert raised.value.code == 2


def test_should_refuse_existing_output_before_loading_and_preserve_failed_attempt(
    tmp_path, monkeypatch
):
    output = tmp_path / "spent.json"
    output.write_bytes(b"original negative result")
    monkeypatch.setattr(cli, "load_saved_fifo_inputs", forbidden)
    with pytest.raises(SystemExit) as raised:
        cli.main(["--saved-fifo-return", "--source-artifact", str(SOURCE), "--output", str(output)])
    assert raised.value.code == 2
    assert output.read_bytes() == b"original negative result"


def test_should_refuse_missing_output_parent_before_loading(tmp_path, monkeypatch):
    output = tmp_path / "missing" / "new.json"
    monkeypatch.setattr(cli, "load_saved_fifo_inputs", forbidden)
    with pytest.raises(SystemExit) as raised:
        cli.main(["--saved-fifo-return", "--source-artifact", str(SOURCE), "--output", str(output)])
    assert raised.value.code == 2
    assert not output.exists()


def test_should_preserve_exclusive_output_when_path_is_created_during_probe(tmp_path, monkeypatch):
    output = tmp_path / "raced.json"
    monkeypatch.setattr(cli, "load_saved_fifo_inputs", lambda path: object())

    def run(inputs, *, max_seconds):
        output.write_bytes(b"other attempt")
        return {"status": "complete"}

    monkeypatch.setattr(cli, "run_saved_fifo_return_pilot", run)
    with pytest.raises(FileExistsError):
        cli.main(["--saved-fifo-return", "--source-artifact", str(SOURCE), "--output", str(output)])
    assert output.read_bytes() == b"other attempt"


def test_should_refuse_oversized_output_without_publishing(tmp_path, monkeypatch):
    output = tmp_path / "oversized.json"
    monkeypatch.setattr(cli, "load_saved_fifo_inputs", lambda path: object())
    monkeypatch.setattr(cli, "MAX_OUTPUT_BYTES", 16)
    monkeypatch.setattr(
        cli, "run_saved_fifo_return_pilot", lambda inputs, **kwargs: {"status": "incomplete"}
    )
    with pytest.raises(ValueError, match="exceeds 32 MiB"):
        cli.main(["--saved-fifo-return", "--source-artifact", str(SOURCE), "--output", str(output)])
    assert not output.exists()
