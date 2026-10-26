"""Pure routing and report contracts; native snapshot proof is separately scoped."""

from copy import deepcopy
import json
from pathlib import Path
import re
import sys
from typing import Any

import pytest

import scripts.run_numpy_managed_capture as cli
import src.adapters.numpy_capture_example as legacy
import src.adapters.numpy_snapshot_capture_example as example


def forbidden(*args, **kwargs):
    pytest.fail("unselected or operational path was reached")


def test_should_describe_fixed_single_graph_controls_without_setup_capture_or_io(monkeypatch):
    monkeypatch.setattr(example, "_setup_snapshot_case", forbidden)
    monkeypatch.setattr(example, "_capture_snapshot_case", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    description = example.describe_snapshot_example()
    assert description["protocol_id"] == "numpy_issued_snapshot_capture_engineering_v1"
    assert description["family"] == "circadian" and description["seed"] == 23
    assert description["geometry"] == {"input_dim": 3, "hidden_dim": 2}
    assert description["features"] == [[1.0, 0.0, 0.0]] and description["targets"] == [[0.0]]
    assert description["learning_rate"] == 0.01
    assert description["inference_steps"] == 2 and description["inference_learning_rate"] == 0.05
    assert description["circadian_config"]["sleep_mode"] == "disabled"
    assert description["replay_retention"] == {"max_examples": 2, "max_bytes": 64}
    assert description["native_training_calls"] == 1
    assert description["capture_calls"] == description["graph_copy_passes"] == 2
    assert description["expected_setup_charge"] == 224
    assert description["expected_retained_bytes"] == 128
    assert description["expected_capture_charge"] == 352
    assert description["data_only"] is True
    assert description["budget"] == {
        "max_training_updates": 2,
        "max_wall_seconds": 1.0,
        "max_replay_examples": None,
        "max_hidden_width": None,
        "max_process_rss_bytes": None,
    }
    assert description["origin_limits"] == {
        "max_live_records": 16,
        "max_records_created": 64,
        "max_invocations": 32,
        "max_metadata_bytes": 128 * 1024,
        "max_age_ticks": 120,
        "max_payload_bytes": 4096,
    }
    assert description["write_limits"] == {
        "max_input_rows": 16,
        "max_retained_snapshots": 4,
        "max_notifications": 64,
    }
    policy = description["retention_policy"]
    assert policy["max_lifetime_ingress_bytes"] == 4096 and policy["max_retention_ticks"] == 120
    assert policy["max_retention_seconds"] == 1000.0
    assert policy["holders"] == {"max_live_holders": 8, "max_lifetime_enrollments": 12}
    assert policy["owned_payload_copies"] == {"max_lifetime_owned_bytes": 4096}
    assert description["capture_limits"] == {
        "records": {"max_records": 64, "max_identifier_bytes": 128},
        "max_nodes": 8192,
        "max_depth": 64,
        "max_array_bytes": 65536,
        "max_dimension": 32,
    }
    assert description["native_limits"] == {
        "max_encoded_bytes": 131072,
        "max_array_bytes": 65536,
        "max_layers": 8,
        "max_dimension": 32,
    }
    assert description["inbox_policy"]["input_dim"] == 3
    assert all(re.fullmatch("[0-9a-f]{64}", value) for value in description["binding"].values())
    limitations = description["limitations"].lower()
    assert all(word in limitations for word in ("authority", "recovery", "scientific"))
    assert len(json.dumps(description, allow_nan=False).encode("utf8")) <= 131072


def test_should_keep_original_describe_schema_and_values_unchanged(monkeypatch):
    before = deepcopy(legacy.describe_example())
    monkeypatch.setattr(example, "_setup_snapshot_case", forbidden)
    monkeypatch.setattr(example, "_capture_snapshot_case", forbidden)
    example.describe_snapshot_example()
    assert legacy.describe_example() == before
    assert before["native_training_calls"] == 0 and before["capture_calls"] == 2
    assert before["families"] == ["backprop", "circadian"]
    assert before["protocol_id"] == "numpy_populated_retired_capture_engineering_v1"


def test_should_forward_exactly_one_borrowed_setup_graph_to_capture_without_rebuilding(monkeypatch):
    graph = object()
    case = {"marker": ["bounded", "data"]}
    calls: list[str] = []

    def setup():
        calls.append("setup")
        return graph

    def capture(value):
        assert value is graph
        calls.append("capture")
        return case

    monkeypatch.setattr(example, "_setup_snapshot_case", setup)
    monkeypatch.setattr(example, "_capture_snapshot_case", capture)
    report = example.run_snapshot_example()
    assert calls == ["setup", "capture"] and report["case"] is case
    assert report["protocol"] == example.describe_snapshot_example()
    assert report["issued_snapshot_capture_verified"] is True
    assert report["full_managed_process_recovery_verified"] is False


class Poison:
    def __init__(self):
        self.reads = 0

    def __getattr__(self, name):
        self.reads += 1
        raise AssertionError("missing setup keys must refuse before native handle access")


class ForeignDict(dict):
    def __iter__(self):
        pytest.fail("foreign mapping hooks must not execute")


@pytest.mark.parametrize("bad", ("none", "object", "list", "missing", "subclass"))
def test_should_refuse_nonexact_or_incomplete_internal_graph_before_operational_access(bad):
    poison = Poison()
    graph: Any = {"runtime": poison}
    if bad == "none":
        graph = None
    elif bad == "object":
        graph = poison
    elif bad == "list":
        graph = []
    elif bad == "subclass":
        graph = ForeignDict(runtime=poison)
    with pytest.raises(ValueError):
        example._capture_snapshot_case(graph)
    assert poison.reads == 0


@pytest.mark.parametrize("snapshot", (False, True))
@pytest.mark.parametrize("action", ("describe", "run"))
def test_should_route_only_explicit_requested_cli_mode_and_emit_semantic_json(
    monkeypatch, capsys, snapshot, action
):
    calls: list[str] = []
    report = {"route": ("snapshot" if snapshot else "legacy", action), "data_only": True}

    def selected():
        calls.append("selected")
        return report

    for name in (
        "describe_example",
        "run_example",
        "describe_snapshot_example",
        "run_snapshot_example",
    ):
        monkeypatch.setattr(cli, name, forbidden)
    name = f"{action}_snapshot_example" if snapshot else f"{action}_example"
    monkeypatch.setattr(cli, name, selected)
    monkeypatch.setattr(
        sys, "argv", ["capture", f"--{action}", *(["--issued-snapshot"] if snapshot else [])]
    )
    assert cli.main() == 0
    actual = json.loads(capsys.readouterr().out)
    expected = json.loads(json.dumps(report, allow_nan=False))  # Explicit JSON tuple/list boundary.
    assert actual == expected and calls == ["selected"]


@pytest.mark.parametrize(
    "arguments",
    (
        [],
        ["--issued-snapshot"],
        ["--describe", "--run", "--issued-snapshot"],
        ["--run", "--issued-snapshot", "--seed", "0"],
        ["--describe", "--issued-snapshot", "--iterations", "3"],
        ["--run", "--issued-snapshot", "--directory", "new"],
    ),
)
def test_should_require_explicit_action_and_refuse_unfrozen_cli_knobs_before_callbacks(
    monkeypatch, arguments
):
    for name in (
        "describe_example",
        "run_example",
        "describe_snapshot_example",
        "run_snapshot_example",
    ):
        monkeypatch.setattr(cli, name, forbidden)
    monkeypatch.setattr(sys, "argv", ["capture", *arguments])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


@pytest.mark.parametrize("snapshot", (False, True))
@pytest.mark.parametrize("number", (float("nan"), float("inf")))
def test_should_reject_nonfinite_report_before_printing_any_output(
    monkeypatch, capsys, snapshot, number
):
    for name in ("run_example", "run_snapshot_example"):
        monkeypatch.setattr(cli, name, lambda: {"unscorable": number})
    monkeypatch.setattr(
        sys, "argv", ["capture", "--run", *(["--issued-snapshot"] if snapshot else [])]
    )
    with pytest.raises(ValueError):
        cli.main()
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("snapshot", (False, True))
@pytest.mark.parametrize("separator", ("\n", "\r\n"))
@pytest.mark.parametrize("offset", (-1, 0, 1))
def test_should_apply_mode_specific_inclusive_utf8_physical_newline_output_bound(
    monkeypatch, capsys, snapshot, separator, offset
):
    maximum = 131072 if snapshot else 524288
    prefix, suffix = '{\n"unicode":"é', '"\n}'
    base = len((prefix + suffix + "\n").replace("\n", separator).encode("utf8"))
    raw = prefix + "x" * (maximum - base + offset) + suffix
    assert len((raw + "\n").replace("\n", separator).encode("utf8")) == maximum + offset
    monkeypatch.setattr(cli.os, "linesep", separator)
    monkeypatch.setattr(cli.json, "dumps", lambda *a, **k: raw)
    monkeypatch.setattr(cli, "describe_example", lambda: {})
    monkeypatch.setattr(cli, "describe_snapshot_example", lambda: {})
    monkeypatch.setattr(
        sys, "argv", ["capture", "--describe", *(["--issued-snapshot"] if snapshot else [])]
    )
    if offset > 0:
        with pytest.raises(ValueError, match="snapshot" if snapshot else "legacy"):
            cli.main()
        assert capsys.readouterr().out == ""
    else:
        assert cli.main() == 0
        assert capsys.readouterr().out == raw + "\n"
