"""Bounded saved-snapshot admission and exclusive assay publication, without runtime."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import pytest

import scripts.run_minigrid_memory_probe as cli
from src.adapters.action_memory_learner import ActionMemoryLearner
from src.core.action_memory import copy_action_memory_snapshot


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "artifacts/runs/minigrid-action-memory-live-20261017/native.json"
)


@pytest.fixture(scope="module")
def saved():
    return json.loads(SOURCE.read_text(encoding="utf-8"))


def write_source(tmp_path, saved):
    path = tmp_path / "source.json"
    path.write_text(json.dumps(saved, allow_nan=False), encoding="utf-8")
    return path


def replace(document: Any, pointer: str, value: Any) -> None:
    tokens = pointer.split("/")[1:]
    current = document
    for token in tokens[:-1]:
        current = current[int(token)] if type(current) is list else current[token]
    if type(current) is list:
        current[int(tokens[-1])] = value
    else:
        current[tokens[-1]] = value


def forbidden(*args, **kwargs):
    pytest.fail("saved-source/describe boundary reached training or assay execution")


def test_should_restore_both_genuine_saved_states_without_training_prediction_or_source_changes(
    saved, monkeypatch
):
    before = SOURCE.read_bytes()
    monkeypatch.setattr(ActionMemoryLearner, "train_batch", forbidden)
    monkeypatch.setattr(ActionMemoryLearner, "predict", forbidden)
    states = cli.load_snapshots(SOURCE)
    assert type(states) is tuple and len(states) == 2
    assert [(s.completed_updates, s.training_examples, len(s.entries)) for s in states] == [
        (0, 0, 0),
        (128, 1152, 54),
    ]
    for state, observation in zip(
        states, (saved["observations"][0], saved["observations"][-1]), strict=True
    ):
        assert json.loads(json.dumps(asdict(state))) == observation["snapshot"]
        assert state.policy.feature_dim == 296 and state.policy.num_classes == 7
        assert state.policy.max_entries == 64 and state.policy.max_payload_bytes == 155136
        assert copy_action_memory_snapshot(state) == state
    object.__setattr__(states[-1].policy, "max_payload_bytes", 2424)
    assert cli.load_snapshots(SOURCE)[-1].policy.max_payload_bytes == 155136
    assert SOURCE.read_bytes() == before
    assert cli.MAX_INPUT_BYTES == cli.MAX_OUTPUT_BYTES == 32 * 1024**2


@pytest.mark.parametrize(
    "pointer,value",
    [
        ("/status", "incomplete"),
        ("/protocol/protocol_id", "minigrid_bounded_public_history_action_memory_v2"),
        ("/protocol/seed", 18),
        ("/protocol/policy/max_payload_bytes", 310272),
        ("/protocol/stream/training_only", False),
        ("/methods/0/method", "bp"),
        ("/methods/0/status", "failed"),
        ("/methods/0/charged_updates", 127),
        ("/methods/0/native_training_examples", 1151),
        ("/methods/0/source_unchanged", False),
        ("/methods/0/actor_unchanged", False),
        ("/observations/0/updates", True),
        ("/observations/4/snapshot/completed_updates", 127),
        ("/observations/4/snapshot/training_examples", 1151),
        ("/observations/4/snapshot/policy/feature_dim", 16),
        ("/observations/4/snapshot/policy/max_payload_bytes", 10752),
        ("/observations/4/snapshot/entries/0/counts/0", True),
        ("/observations/4/snapshot/entries/0/features", [0.0] * 295),
        ("/observations/4/entries", 53),
        ("/observations/4/logical_payload_bytes", 130895),
        ("/training_sources/0/role", "final_test"),
    ],
)
def test_should_refuse_changed_source_protocol_roles_geometry_counters_or_snapshot_fields(
    saved, tmp_path, pointer, value
):
    document = deepcopy(saved)
    replace(document, pointer, value)
    source = write_source(tmp_path, document)
    before = source.read_bytes()
    with pytest.raises(ValueError):
        cli.load_snapshots(source)
    assert source.read_bytes() == before


@pytest.mark.parametrize(
    "fault",
    [
        "extra_snapshot_field",
        "duplicate_observation",
        "missing_observation",
        "extra_method",
        "missing_snapshot",
        "extra_entry_field",
    ],
)
def test_should_refuse_ambiguous_or_incomplete_saved_snapshot_sets(saved, tmp_path, fault):
    document = deepcopy(saved)
    if fault == "extra_snapshot_field":
        document["observations"][-1]["snapshot"]["extra"] = 1
    elif fault == "duplicate_observation":
        document["observations"][-1] = deepcopy(document["observations"][-2])
    elif fault == "missing_observation":
        document["observations"].pop(2)
    elif fault == "extra_method":
        document["methods"].append(deepcopy(document["methods"][0]))
    elif fault == "missing_snapshot":
        del document["observations"][-1]["snapshot"]
    else:
        document["observations"][-1]["snapshot"]["entries"][0]["extra"] = 1
    with pytest.raises(ValueError):
        cli.load_snapshots(write_source(tmp_path, document))


@pytest.mark.parametrize(
    "raw",
    [b'{"x":NaN}', b'{"x":Infinity}', b'{"x":1e999}', b'{"x":1,"x":2}', b"\xff", b'{"broken":'],
)
def test_should_refuse_invalid_json_nonfinite_values_and_duplicate_fields(tmp_path, raw):
    source = tmp_path / "malformed.json"
    source.write_bytes(raw)
    with pytest.raises(ValueError):
        cli.load_snapshots(source)
    assert source.read_bytes() == raw


def test_should_bound_input_before_parsing(tmp_path, monkeypatch):
    source = tmp_path / "oversized.json"
    source.write_bytes(b" " * 65)
    monkeypatch.setattr(cli, "MAX_INPUT_BYTES", 64)
    with pytest.raises(ValueError):
        cli.load_snapshots(source)


def test_should_describe_without_loading_source_or_running_any_assay(monkeypatch, capsys):
    expected = {"protocol_id": "description-only", "probe_updates": 32}
    monkeypatch.setattr(cli, "describe_protocol", lambda: deepcopy(expected))
    monkeypatch.setattr(cli, "load_snapshots", forbidden)
    monkeypatch.setattr(cli, "run_pilot", forbidden)
    assert cli.main(["--describe"]) == 0
    assert json.loads(capsys.readouterr().out) == expected


@pytest.mark.parametrize(
    "arguments",
    [[], ["--describe", "--source-artifact", "x.json"], ["--describe", "--output", "x.json"]],
)
def test_should_refuse_missing_source_or_mixed_describe_arguments_before_loading(
    monkeypatch, arguments
):
    monkeypatch.setattr(cli, "load_snapshots", forbidden)
    monkeypatch.setattr(cli, "run_pilot", forbidden)
    with pytest.raises(SystemExit) as raised:
        cli.main(arguments)
    assert raised.value.code == 2


@pytest.mark.parametrize("status,exit_code", [("complete", 0), ("incomplete", 2)])
def test_should_publish_complete_or_negative_report_once_and_preserve_source(
    tmp_path, monkeypatch, capsys, status, exit_code
):
    output = tmp_path / "assay.json"
    before = SOURCE.read_bytes()
    states: Any = (object(), object())
    calls = []
    monkeypatch.setattr(cli, "load_snapshots", lambda path: states)

    def run(received):
        calls.append(received)
        return {
            "status": status,
            "cohorts": [{"status": "complete"}, {"status": "failed", "charged_updates": 1}],
            "reason": "capacity negative preserved",
        }

    monkeypatch.setattr(cli, "run_pilot", run)
    assert cli.main(["--source-artifact", str(SOURCE), "--output", str(output)]) == exit_code
    assert calls == [states]
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == status and report["cohorts"][1]["charged_updates"] == 1
    assert report["reason"] == "capacity negative preserved"
    assert report["source_artifact"] == str(SOURCE)
    assert json.loads(capsys.readouterr().out) == report
    assert SOURCE.read_bytes() == before


def test_should_refuse_existing_output_before_loading_or_running(tmp_path, monkeypatch):
    output = tmp_path / "existing.json"
    output.write_bytes(b"unrelated user data")
    monkeypatch.setattr(cli, "load_snapshots", forbidden)
    monkeypatch.setattr(cli, "run_pilot", forbidden)
    with pytest.raises(SystemExit) as raised:
        cli.main(["--source-artifact", str(SOURCE), "--output", str(output)])
    assert raised.value.code == 2 and output.read_bytes() == b"unrelated user data"


def test_should_refuse_a_racing_output_without_overwriting_it(tmp_path, monkeypatch):
    output = tmp_path / "raced.json"
    monkeypatch.setattr(cli, "load_snapshots", lambda path: ())

    def run(states):
        output.write_bytes(b"created while assay was running")
        return {"status": "incomplete", "reason": "negative"}

    monkeypatch.setattr(cli, "run_pilot", run)
    with pytest.raises(FileExistsError):
        cli.main(["--source-artifact", str(SOURCE), "--output", str(output)])
    assert output.read_bytes() == b"created while assay was running"


def test_should_refuse_source_collision_and_oversized_output_without_publication(
    tmp_path, monkeypatch
):
    before = SOURCE.read_bytes()
    monkeypatch.setattr(cli, "load_snapshots", forbidden)
    monkeypatch.setattr(cli, "run_pilot", forbidden)
    with pytest.raises(SystemExit):
        cli.main(["--source-artifact", str(SOURCE), "--output", str(SOURCE)])
    assert SOURCE.read_bytes() == before
    output = tmp_path / "oversized.json"
    monkeypatch.setattr(cli, "load_snapshots", lambda path: ())
    monkeypatch.setattr(
        cli, "run_pilot", lambda states: {"status": "incomplete", "text": "x" * 1000}
    )
    monkeypatch.setattr(cli, "MAX_OUTPUT_BYTES", 64)
    with pytest.raises(ValueError, match="output"):
        cli.main(["--source-artifact", str(SOURCE), "--output", str(output)])
    assert not output.exists()
