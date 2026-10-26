"""FIFO36 is explicit; prior raw/refusal-invariant CLI calls remain unchanged."""

import json
import sys

import pytest

import scripts.run_coupled_attribution as cli
from src.adapters import minigrid_action_memory_pilot as pilot
from src.core.minigrid_invariant_encoding import INVARIANT_HISTORY_ENCODING_ID
from test_torch_feature_rehearsal_pilot import forbidden


@pytest.mark.parametrize("describe", [True, False])
@pytest.mark.parametrize("mode", ["raw", "invariant", "fifo"])
def test_should_forward_only_the_selected_existing_or_new_recipe(
    monkeypatch, capsys, describe, mode
):
    calls = []

    def selected(**options):
        calls.append(options)
        return dict(status="complete")

    monkeypatch.setattr(
        cli, "describe_minigrid_action_memory_protocol", selected if describe else forbidden
    )
    monkeypatch.setattr(
        cli, "run_minigrid_action_memory_pilot", forbidden if describe else selected
    )
    flag = {"raw": [], "invariant": ["--invariant-memory"], "fifo": ["--invariant-fifo-memory"]}[
        mode
    ]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pilot",
            "--scenario",
            "minigrid-action-memory",
            *flag,
            *(["--describe"] if describe else []),
        ],
    )
    cli.main()
    assert calls == [
        {
            "raw": {},
            "invariant": dict(invariant_memory=True),
            "fifo": dict(invariant_fifo_memory=True),
        }[mode]
    ]
    assert json.loads(capsys.readouterr().out)["status"] == "complete"


@pytest.mark.parametrize(
    "arguments",
    [
        ["--scenario", "single", "--invariant-fifo-memory"],
        ["--scenario", "torch-minigrid-history-learnability", "--invariant-fifo-memory"],
        ["--scenario", "minigrid-action-memory", "--invariant-fifo-memory", "--invariant-memory"],
    ],
)
def test_should_refuse_other_scenarios_and_mutually_exclusive_memory_controls(
    monkeypatch, arguments
):
    for name in (
        "describe_minigrid_action_memory_protocol",
        "run_minigrid_action_memory_pilot",
        "describe_minigrid_learnability_protocol",
        "run_minigrid_learnability_pilot",
    ):
        monkeypatch.setattr(cli, name, forbidden)
    monkeypatch.setattr(sys, "argv", ["pilot", *arguments, "--describe"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


@pytest.mark.parametrize("options", [{}, dict(invariant_memory=True)])
def test_should_preserve_previous_descriptors_when_fifo_option_is_false(options):
    assert pilot.describe_protocol(**options) == pilot.describe_protocol(
        **options, invariant_fifo_memory=False
    )


def test_should_refuse_both_memory_options_before_any_resource_port(monkeypatch):
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    for operation in (pilot.describe_protocol, pilot.run_pilot):
        with pytest.raises(ValueError):
            operation(invariant_memory=True, invariant_fifo_memory=True)


def test_should_describe_fixed_capacity_and_unchanged_recurring_exposure_without_ports(monkeypatch):
    for name in ("ProcessRssSampler", "_invariant_fifo_learner", "_invariant_learner"):
        monkeypatch.setattr(pilot, name, forbidden)
    raw = pilot.describe_protocol()
    fifo = pilot.describe_protocol(invariant_fifo_memory=True)
    assert fifo["protocol_id"] == "minigrid_bounded_invariant_fifo36_history_action_memory_v1"
    assert fifo["methods"] == [pilot.INVARIANT_FIFO_METHOD]
    assert fifo["policy"]["encoding_id"] == INVARIANT_HISTORY_ENCODING_ID
    native = fifo["policy"]["native_policy"]
    assert native["method"] == "observation_memory_fifo"
    assert native["max_entries"] == 36 and native["max_payload_bytes"] == 87264
    for key in ("task_sequence", "checkpoint_updates", "stream", "limits"):
        assert fifo[key] == raw[key]
    for key in (
        "storage_updates",
        "training_rows",
        "raw_score_forwards",
        "raw_score_rows",
        "live_episodes",
    ):
        assert fifo["planned_work"][key] == raw["planned_work"][key]
    assert fifo["trainable_parameters"] is None


def test_should_publish_negative_fifo_result_and_exit_two(monkeypatch, capsys):
    negative = dict(
        status="incomplete", reason="engineering native receipt unavailable", observations=[]
    )
    monkeypatch.setattr(cli, "run_minigrid_action_memory_pilot", lambda **options: negative)
    monkeypatch.setattr(
        sys, "argv", ["pilot", "--scenario", "minigrid-action-memory", "--invariant-fifo-memory"]
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert json.loads(capsys.readouterr().out) == negative
