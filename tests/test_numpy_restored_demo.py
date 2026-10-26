"""Pure restored-demo contracts; genuine replay proof remains a separate gate."""

from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any

import pytest

import scripts.run_numpy_managed_capture as cli
import src.adapters.numpy_capture_example as legacy
import src.adapters.numpy_restored_capture_example as example
import src.adapters.numpy_snapshot_capture_example as snapshot


HANDLERS = (
    "describe_example",
    "run_example",
    "describe_snapshot_example",
    "run_snapshot_example",
    "describe_restored_example",
    "run_restored_example",
)
GRAPH_KEYS = (
    "owner",
    "life",
    "runtime",
    "ledger",
    "controller",
    "manager",
    "token",
    "pending",
    "view",
    "snapshot",
    "source",
    "shared",
    "actor",
    "budget",
    "clock",
    "gate",
    "callbacks",
    "codec",
    "features",
    "targets",
    "wall_clock",
    "builder",
    "old",
    "new",
    "before_publication",
    "copy_before_publication",
)


def forbidden(*args, **kwargs):
    pytest.fail("an unselected or operational path was reached")


def test_should_describe_fixed_restored_data_controls_without_any_setup_capture_or_io(monkeypatch):
    monkeypatch.setattr(example, "_setup_restored_case", forbidden)
    monkeypatch.setattr(example, "_capture_restored_case", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    description = example.describe_restored_example()
    expected = {
        "protocol_id": "numpy_restored_replay_capture_engineering_v1",
        "family": "circadian",
        "seed": 23,
        "geometry": {"input_dim": 3, "hidden_dim": 2},
        "features": [[1.0, 0.0, 0.0]],
        "targets": [[0.0]],
        "learning_rate": 0.01,
        "inference_steps": 2,
        "inference_learning_rate": 0.05,
        "retain_capture_history": True,
        "native_training_calls": 1,
        "native_restore_calls": 1,
        "checkpoint_issues": 1,
        "checkpoint_restores": 1,
        "handoffs": 1,
        "capture_calls": 2,
        "graph_copy_passes": 2,
        "expected_setup_charge": 608,
        "expected_retained_bytes": 128,
        "expected_capture_charge": 736,
        "data_only": True,
        "publication_metadata_charge_rule": "canonical UTF-8 JSON(metadata) bytes + 1024",
    }
    assert {key: description[key] for key in expected} == expected
    assert description["circadian_config"]["sleep_mode"] == "disabled"
    assert description["replay_retention"] == {"max_examples": 2, "max_bytes": 64}
    assert description["origin_limits"] == {
        "max_live_records": 16,
        "max_records_created": 64,
        "max_invocations": 32,
        "max_metadata_bytes": 131072,
        "max_age_ticks": 120,
        "max_payload_bytes": 4096,
    }
    assert description["write_limits"] == {
        "max_input_rows": 16,
        "max_retained_snapshots": 4,
        "max_notifications": 64,
    }
    assert description["budget"] == {
        "max_training_updates": 2,
        "max_wall_seconds": 1.0,
        "max_replay_examples": None,
        "max_hidden_width": None,
        "max_process_rss_bytes": None,
    }
    policy = description["retention_policy"]
    assert policy["max_lifetime_ingress_bytes"] == 4096
    assert policy["max_retention_ticks"] == 120 and policy["max_retention_seconds"] == 1000.0
    assert policy["holders"] == {"max_live_holders": 8, "max_lifetime_enrollments": 12}
    assert policy["owned_payload_copies"] == {"max_lifetime_owned_bytes": 4096}
    assert description["telemetry"] == {
        "source": "injected",
        "wall_seconds": 0.0,
        "physical_rss": False,
        "workers": 0,
    }
    assert all(
        word in description["limitations"].lower()
        for word in ("authority", "recovery", "scientific")
    )
    assert len(json.dumps(description, allow_nan=False).encode("utf8")) <= 131072


def test_should_preserve_both_previous_descriptions_and_shared_scientific_controls(monkeypatch):
    previous = deepcopy(legacy.describe_example()), deepcopy(snapshot.describe_snapshot_example())
    monkeypatch.setattr(example, "_setup_restored_case", forbidden)
    monkeypatch.setattr(example, "_capture_restored_case", forbidden)
    restored = example.describe_restored_example()
    assert legacy.describe_example() == previous[0]
    assert snapshot.describe_snapshot_example() == previous[1]
    for name in (
        "circadian_config",
        "replay_retention",
        "retention_policy",
        "origin_limits",
        "write_limits",
        "budget",
        "capture_limits",
        "native_limits",
        "inbox_policy",
        "binding",
        "telemetry",
    ):
        assert restored[name] == previous[1][name]
    assert previous[0]["native_training_calls"] == 0
    assert previous[1]["expected_setup_charge"] == 224


def test_should_forward_one_original_setup_graph_once_and_report_only_the_scoped_result(
    monkeypatch,
):
    original = object()
    case = {"marker": ["bounded", "data"]}
    calls: list[str] = []

    def setup():
        calls.append("setup")
        return original

    def capture(value):
        assert value is original
        calls.append("capture")
        return case

    monkeypatch.setattr(example, "_setup_restored_case", setup)
    monkeypatch.setattr(example, "_capture_restored_case", capture)
    report = example.run_restored_example()
    assert calls == ["setup", "capture"] and report["case"] is case
    assert report["protocol"] == example.describe_restored_example()
    assert report["restored_replay_capture_verified"] is True
    assert report["full_managed_process_recovery_verified"] is False


class Poison:
    def __init__(self):
        self.reads = 0

    def __getattr__(self, name):
        self.reads += 1
        raise AssertionError("invalid graph reached an operational handle")


class ForeignDict(dict):
    def keys(self):
        pytest.fail("foreign mapping hooks were reached")


@pytest.mark.parametrize("bad", ("none", "object", "list", "missing", "subclass", "extra"))
def test_should_refuse_nonexact_incomplete_or_extra_internal_graph_before_handle_access(bad):
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
    elif bad == "extra":
        graph = dict.fromkeys(GRAPH_KEYS, poison)
        graph["foreign"] = poison
    with pytest.raises(ValueError):
        example._capture_restored_case(graph)
    assert poison.reads == 0


@pytest.mark.parametrize("mode", ("legacy", "snapshot", "restored"))
@pytest.mark.parametrize("action", ("describe", "run"))
def test_should_route_exactly_one_of_six_explicit_cli_handlers(monkeypatch, capsys, mode, action):
    calls: list[str] = []
    report = {"route": (mode, action), "data_only": True}

    def selected():
        calls.append("selected")
        return report

    for name in HANDLERS:
        monkeypatch.setattr(cli, name, forbidden)
    suffix = "example" if mode == "legacy" else f"{mode}_example"
    monkeypatch.setattr(cli, f"{action}_{suffix}", selected)
    flag = (
        []
        if mode == "legacy"
        else ["--issued-snapshot" if mode == "snapshot" else "--restored-replay"]
    )
    monkeypatch.setattr(sys, "argv", ["capture", f"--{action}", *flag])
    assert cli.main() == 0
    # Ordinary JSON changes tuples to lists; compare the actual boundary values.
    assert json.loads(capsys.readouterr().out) == json.loads(json.dumps(report, allow_nan=False))
    assert calls == ["selected"]


@pytest.mark.parametrize(
    "arguments",
    (
        [],
        ["--restored-replay"],
        ["--describe", "--issued-snapshot", "--restored-replay"],
        ["--describe", "--run", "--restored-replay"],
        ["--run", "--restored-replay", "--seed", "0"],
        ["--describe", "--restored-replay", "--iterations", "3"],
        ["--run", "--restored-replay", "--directory", "new"],
        ["--run", "--restored-replay", "--learning-rate", ".2"],
        ["--run", "--restored-replay", "--copies", "0"],
        ["--run", "--restored-replay", "--restore", "false"],
        ["--run", "--restored-replay", "--allow-foreign"],
    ),
)
def test_should_require_an_action_and_refuse_conflicts_or_unfrozen_knobs_before_callbacks(
    monkeypatch, arguments
):
    for name in HANDLERS:
        monkeypatch.setattr(cli, name, forbidden)
    monkeypatch.setattr(sys, "argv", ["capture", *arguments])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


@pytest.mark.parametrize("value", (float("nan"), float("inf")))
def test_should_refuse_nonfinite_restored_report_without_partial_stdout(monkeypatch, capsys, value):
    monkeypatch.setattr(cli, "run_restored_example", lambda: {"invalid": value})
    monkeypatch.setattr(sys, "argv", ["capture", "--run", "--restored-replay"])
    with pytest.raises(ValueError):
        cli.main()
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("separator", ("\n", "\r\n"))
@pytest.mark.parametrize("offset", (-1, 0, 1))
def test_should_bound_actual_utf8_physical_newline_output_inclusively(
    monkeypatch, capsys, separator, offset
):
    maximum = 131072
    prefix, suffix = '{\n"unicode":"é', '"\n}'
    baseline = len((prefix + suffix + "\n").replace("\n", separator).encode("utf8"))
    raw = prefix + "x" * (maximum - baseline + offset) + suffix
    assert len((raw + "\n").replace("\n", separator).encode("utf8")) == maximum + offset
    monkeypatch.setattr(cli.os, "linesep", separator)
    monkeypatch.setattr(cli.json, "dumps", lambda *a, **k: raw)
    monkeypatch.setattr(cli, "describe_restored_example", lambda: {})
    monkeypatch.setattr(sys, "argv", ["capture", "--describe", "--restored-replay"])
    if offset > 0:
        with pytest.raises(ValueError, match="restored replay.*bounded output allowance"):
            cli.main()
        assert capsys.readouterr().out == ""
    else:
        assert cli.main() == 0
        assert capsys.readouterr().out == raw + "\n"
