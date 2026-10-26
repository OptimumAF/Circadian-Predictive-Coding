"""Explicit opt-in dispatch; no neural selector or default recipe change."""

import sys

import pytest

import scripts.run_coupled_attribution as cli
from src.adapters import minigrid_learnability as pilot
from test_minigrid_memory_control import FLAGS
from test_minigrid_memory_control_cli import FLAGS as CLI_FLAGS
from test_torch_feature_rehearsal_pilot import forbidden


@pytest.mark.parametrize("describe", [True, False])
def test_should_dispatch_only_explicit_count_endpoint_games(monkeypatch, describe):
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
            "--probe-rollouts",
            *(["--describe"] if describe else []),
        ],
    )
    cli.main()
    assert calls == [dict(**FLAGS, probe_rollouts=True)]


@pytest.mark.parametrize("extra", [[], ["--qualification-method", "bp"], ["--train-a-rollout"]])
def test_should_refuse_non_count_rollouts_before_dispatch(monkeypatch, extra):
    monkeypatch.setattr(cli, "describe_minigrid_learnability_protocol", forbidden)
    monkeypatch.setattr(cli, "run_minigrid_learnability_pilot", forbidden)
    monkeypatch.setattr(
        sys, "argv", ["pilot", *CLI_FLAGS, "--probe-rollouts", *extra, "--describe"]
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_should_keep_disabled_count_protocol_exact_and_describe_new_game_dose():
    assert pilot.describe_protocol(**FLAGS) == pilot.describe_protocol(
        **FLAGS, probe_rollouts=False
    )
    protocol = pilot.describe_protocol(**FLAGS, probe_rollouts=True)
    assert protocol["planned_work"]["learner_live_episodes"] == 16
    assert protocol["planned_work"]["maximum_live_prediction_forwards"] == 256
    assert protocol["planned_work"]["score_forwards"] == 94
    assert protocol["planned_work"]["native_train_calls"] == 192
