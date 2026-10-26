"""Pure successive CLI contracts; replay authority is tested independently."""

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
from test_numpy_restored_demo import GRAPH_KEYS


HANDLERS = (
    "describe_example",
    "run_example",
    "describe_snapshot_example",
    "run_snapshot_example",
    "describe_restored_example",
    "run_restored_example",
    "describe_successive_example",
    "run_successive_example",
)


def forbidden(*args, **kwargs):
    pytest.fail("an unselected or operational path was reached")


def test_should_describe_fixed_successive_controls_without_setup_capture_or_io(monkeypatch):
    monkeypatch.setattr(example, "_setup_restored_case", forbidden)
    monkeypatch.setattr(example, "_setup_successive_case", forbidden)
    monkeypatch.setattr(example, "_capture_restored_case", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    description = example.describe_successive_example()
    expected = {
        "protocol_id": "numpy_successive_replay_capture_engineering_v1",
        "family": "circadian",
        "seed": 23,
        "geometry": {"input_dim": 3, "hidden_dim": 2},
        "features": [[1.0, 0.0, 0.0]],
        "targets": [[0.0]],
        "learning_rate": 0.01,
        "inference_steps": 2,
        "inference_learning_rate": 0.05,
        "retain_capture_history": True,
        "retain_inbox_origins": True,
        "native_training_calls": 3,
        "paid_paused_polls": 1,
        "native_restore_calls": 1,
        "checkpoint_issues": 1,
        "checkpoint_restores": 1,
        "handoffs": 1,
        "capture_calls": 2,
        "graph_copy_passes": 2,
        "expected_setup_charge": 800,
        "expected_retained_bytes": 256,
        "expected_capture_charge": 1056,
        "data_only": True,
        "publication_metadata_charge_rule": "canonical UTF-8 JSON(metadata) bytes + 1024",
        "current_success_metadata_charge_rule": "canonical UTF-8 JSON(metadata) bytes + 1024",
        "zero_work_poll_metadata_charge_rule": "canonical UTF-8 JSON(metadata) bytes + 1024",
    }
    assert {name: description[name] for name in expected} == expected
    assert description["circadian_config"]["sleep_mode"] == "disabled"
    assert description["replay_retention"] == {"max_examples": 2, "max_bytes": 64}
    assert description["update_sequence"] == [
        {"features": [[1.0, 0.0, 0.0]], "targets": [[0.0]], "observed_at": 1, "arrived_at": 3},
        {
            "features": [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "targets": [[1.0], [0.0]],
            "observed_at": 4,
            "arrived_at": 6,
        },
        {"features": [[1.0, 1.0, 0.0]], "targets": [[1.0]], "observed_at": 8, "arrived_at": 10},
    ]
    assert description["origin_limits"] == {
        "max_live_records": 32,
        "max_records_created": 64,
        "max_invocations": 32,
        "max_metadata_bytes": 262144,
        "max_age_ticks": 120,
        "max_payload_bytes": 4096,
    }
    assert description["budget"] == {
        "max_training_updates": 3,
        "max_wall_seconds": 1.0,
        "max_replay_examples": None,
        "max_hidden_width": None,
        "max_process_rss_bytes": None,
    }
    assert description["write_limits"] == {
        "max_input_rows": 16,
        "max_retained_snapshots": 4,
        "max_notifications": 64,
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


def test_should_leave_all_previous_descriptions_and_shared_controls_unchanged(monkeypatch):
    before = tuple(
        deepcopy(describe())
        for describe in (
            legacy.describe_example,
            snapshot.describe_snapshot_example,
            example.describe_restored_example,
        )
    )
    monkeypatch.setattr(example, "_setup_successive_case", forbidden)
    monkeypatch.setattr(example, "_capture_restored_case", forbidden)
    successive = example.describe_successive_example()
    assert (
        tuple(
            describe()
            for describe in (
                legacy.describe_example,
                snapshot.describe_snapshot_example,
                example.describe_restored_example,
            )
        )
        == before
    )
    for name in (
        "circadian_config",
        "replay_retention",
        "retention_policy",
        "write_limits",
        "capture_limits",
        "native_limits",
        "inbox_policy",
        "binding",
        "telemetry",
    ):
        assert successive[name] == before[2][name]
    assert before[0]["native_training_calls"] == 0
    assert before[1]["expected_setup_charge"] == 224
    assert before[2]["expected_setup_charge"] == 608
    assert before[2]["origin_limits"]["max_metadata_bytes"] == 131072
    assert before[2]["budget"]["max_training_updates"] == 2


def test_should_forward_one_original_graph_once_with_the_explicit_successive_keyword(monkeypatch):
    graph = object()
    case, protocol = {"marker": ["bounded", "data"]}, {"marker": "description"}
    calls: list[str] = []

    def setup():
        calls.append("setup")
        return graph

    def capture(value, *, successive=False):
        assert value is graph and successive is True
        calls.append("capture")
        return case

    monkeypatch.setattr(example, "_setup_restored_case", forbidden)
    monkeypatch.setattr(example, "_setup_successive_case", setup)
    monkeypatch.setattr(example, "_capture_restored_case", capture)
    monkeypatch.setattr(example, "describe_successive_example", lambda: protocol)
    report = example.run_successive_example()
    assert calls == ["setup", "capture"]
    assert report == {
        "protocol": protocol,
        "case": case,
        "successive_replay_capture_verified": True,
        "full_managed_process_recovery_verified": False,
    }
    assert report["protocol"] is protocol and report["case"] is case


class Poison:
    def __init__(self):
        self.reads = 0

    def __getattr__(self, name):
        self.reads += 1
        raise AssertionError("invalid graph reached an operational handle")


class ForeignDict(dict):
    def keys(self):
        pytest.fail("foreign mapping hooks were reached")


@pytest.mark.parametrize(
    "bad",
    (
        "none",
        "object",
        "subclass",
        "missing_continuation",
        "extra",
        "default_with_continuation",
        "flag_none",
        "flag_int",
    ),
)
def test_should_refuse_wrong_graphs_or_nonboolean_mode_before_operational_handles(bad):
    poison = Poison()
    graph: Any = dict.fromkeys((*GRAPH_KEYS, "continuation"), poison)
    successive: Any = True
    if bad == "none":
        graph = None
    elif bad == "object":
        graph = poison
    elif bad == "subclass":
        graph = ForeignDict(graph)
    elif bad == "missing_continuation":
        del graph["continuation"]
    elif bad == "extra":
        graph["foreign"] = poison
    elif bad == "default_with_continuation":
        successive = False
    elif bad == "flag_none":
        successive = None
    else:
        successive = 1
    with pytest.raises(ValueError):
        example._capture_restored_case(graph, successive=successive)
    assert poison.reads == 0


@pytest.mark.parametrize("mode", ("legacy", "snapshot", "restored", "successive"))
@pytest.mark.parametrize("action", ("describe", "run"))
def test_should_route_only_the_selected_handler_for_all_eight_explicit_modes(
    monkeypatch,
    capsys,
    mode,
    action,
):
    calls: list[str] = []
    report = {"route": (mode, action), "data_only": True}

    def selected():
        calls.append("selected")
        return report

    for name in HANDLERS:
        monkeypatch.setattr(cli, name, forbidden)
    suffix = "example" if mode == "legacy" else f"{mode}_example"
    monkeypatch.setattr(cli, f"{action}_{suffix}", selected)
    flags = {
        "legacy": [],
        "snapshot": ["--issued-snapshot"],
        "restored": ["--restored-replay"],
        "successive": ["--successive-replay"],
    }
    monkeypatch.setattr(sys, "argv", ["capture", f"--{action}", *flags[mode]])
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out) == json.loads(json.dumps(report, allow_nan=False))
    assert calls == ["selected"]


@pytest.mark.parametrize(
    "arguments",
    (
        ["--successive-replay"],
        ["--describe", "--run", "--successive-replay"],
        ["--describe", "--successive-replay", "--issued-snapshot"],
        ["--describe", "--successive-replay", "--restored-replay"],
        ["--run", "--successive-replay", "--seed", "0"],
        ["--run", "--successive-replay", "--iterations", "3"],
        ["--run", "--successive-replay", "--directory", "new"],
        ["--run", "--successive-replay", "--learning-rate", ".2"],
        ["--run", "--successive-replay", "--copies", "0"],
        ["--run", "--successive-replay", "--updates", "4"],
        ["--run", "--successive-replay", "--allow-foreign"],
    ),
)
def test_should_refuse_missing_actions_conflicts_and_unfrozen_knobs_before_callbacks(
    monkeypatch,
    arguments,
):
    for name in HANDLERS:
        monkeypatch.setattr(cli, name, forbidden)
    monkeypatch.setattr(sys, "argv", ["capture", *arguments])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
def test_should_refuse_nonfinite_successive_reports_without_partial_stdout(
    monkeypatch, capsys, value
):
    monkeypatch.setattr(cli, "run_successive_example", lambda: {"invalid": value})
    monkeypatch.setattr(sys, "argv", ["capture", "--run", "--successive-replay"])
    with pytest.raises(ValueError):
        cli.main()
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("separator", ("\n", "\r\n"))
@pytest.mark.parametrize("offset", (-1, 0, 1))
def test_should_bound_successive_utf8_output_including_physical_newline(
    monkeypatch,
    capsys,
    separator,
    offset,
):
    maximum = 131072
    prefix, suffix = '{\n"unicode":"é', '"\n}'
    baseline = len((prefix + suffix + "\n").replace("\n", separator).encode("utf8"))
    raw = prefix + "x" * (maximum - baseline + offset) + suffix
    assert len((raw + "\n").replace("\n", separator).encode("utf8")) == maximum + offset
    monkeypatch.setattr(cli.os, "linesep", separator)
    monkeypatch.setattr(cli.json, "dumps", lambda *a, **k: raw)
    monkeypatch.setattr(cli, "describe_successive_example", lambda: {})
    monkeypatch.setattr(sys, "argv", ["capture", "--describe", "--successive-replay"])
    if offset > 0:
        with pytest.raises(ValueError, match="successive replay.*bounded output allowance"):
            cli.main()
        assert capsys.readouterr().out == ""
    else:
        assert cli.main() == 0
        assert capsys.readouterr().out == raw + "\n"
