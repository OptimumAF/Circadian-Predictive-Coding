"""Explicit post-training game dispatch; previews allocate no learner ports."""

import json
import sys

import pytest

import scripts.run_coupled_attribution as cli
from src.adapters import minigrid_learnability as pilot
from test_minigrid_memory_control import FLAGS
from test_minigrid_memory_control_cli import FLAGS as CLI_FLAGS
from test_torch_feature_rehearsal_pilot import forbidden


@pytest.mark.parametrize("describe", [True, False])
@pytest.mark.parametrize("enabled", [True, False])
def test_should_forward_only_explicit_deferred_games(monkeypatch, capsys, describe, enabled):
    calls = []

    def selected(**options):
        calls.append(options)
        return dict(status="complete")

    monkeypatch.setattr(
        cli, "describe_minigrid_learnability_protocol", selected if describe else forbidden
    )
    monkeypatch.setattr(cli, "run_minigrid_learnability_pilot", forbidden if describe else selected)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pilot",
            *CLI_FLAGS,
            "--memory-control",
            *(["--deferred-probe-rollouts"] if enabled else []),
            *(["--describe"] if describe else []),
        ],
    )
    cli.main()
    expected = dict(FLAGS)
    if enabled:
        expected["deferred_probe_rollouts"] = True
    assert calls == [expected]
    assert json.loads(capsys.readouterr().out)["status"] == "complete"


@pytest.mark.parametrize(
    "arguments",
    [
        [*CLI_FLAGS, "--deferred-probe-rollouts"],
        [*CLI_FLAGS, "--memory-control", "--deferred-probe-rollouts", "--probe-rollouts"],
        [
            *CLI_FLAGS,
            "--memory-control",
            "--deferred-probe-rollouts",
            "--qualification-method",
            "bp",
        ],
        [*CLI_FLAGS, "--memory-control", "--deferred-probe-rollouts", "--train-a-rollout"],
        ["--scenario", "single", "--deferred-probe-rollouts"],
        [
            *(arg for arg in CLI_FLAGS if arg != "--return-to-a"),
            "--memory-control",
            "--deferred-probe-rollouts",
        ],
    ],
)
def test_should_refuse_incompatible_cli_modes_before_dispatch(monkeypatch, arguments):
    monkeypatch.setattr(cli, "describe_minigrid_learnability_protocol", forbidden)
    monkeypatch.setattr(cli, "run_minigrid_learnability_pilot", forbidden)
    monkeypatch.setattr(sys, "argv", ["pilot", *arguments, "--describe"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_should_keep_negative_game_result_and_exit_two(monkeypatch, capsys):
    negative = dict(
        status="incomplete",
        reason="original phase deadline expired",
        deferred_probe_rollouts=dict(status="failed", entry_validated=True, exit_validated=False),
    )
    monkeypatch.setattr(cli, "run_minigrid_learnability_pilot", lambda **options: negative)
    monkeypatch.setattr(
        sys, "argv", ["pilot", *CLI_FLAGS, "--memory-control", "--deferred-probe-rollouts"]
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert json.loads(capsys.readouterr().out) == negative


@pytest.mark.parametrize("options", [{}, FLAGS, dict(FLAGS, probe_rollouts=True)])
def test_should_preserve_disabled_protocol_and_omit_new_descriptor_fields(options):
    assert pilot.describe_protocol(**options) == pilot.describe_protocol(
        **options, deferred_probe_rollouts=False
    )


def test_should_describe_same_training_and_bounded_post_training_games_without_resources(
    monkeypatch,
):
    for name in ("ProcessRssSampler", "require_torch", "_history_learner", "MiniGridKeyDoor"):
        monkeypatch.setattr(pilot, name, forbidden)
    baseline = pilot.describe_protocol(**FLAGS)
    actual = pilot.describe_protocol(**FLAGS, deferred_probe_rollouts=True)
    for field in ("native_train_calls", "training_rows", "score_forwards", "score_rows"):
        assert actual["planned_work"][field] == baseline["planned_work"][field]
    assert actual["planned_work"]["learner_live_episodes"] == 16
    assert actual["planned_work"]["maximum_live_prediction_forwards"] == 256
    assert actual["methods"] == baseline["methods"]
