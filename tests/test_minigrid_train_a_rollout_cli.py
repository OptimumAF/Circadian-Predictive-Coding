"""Explicit known-TRAIN live readout uses the existing learnability CLI."""

import json
import sys

import pytest

import scripts.run_coupled_attribution as cli
from test_torch_feature_rehearsal_pilot import forbidden


@pytest.mark.parametrize("method", ["bp", "pc", "neutral_cpc", "gated_cpc"])
@pytest.mark.parametrize("describe", [True, False])
def test_should_dispatch_opt_in_live_games_without_changing_the_recipe(
    monkeypatch, capsys, method, describe
):
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
            "--scenario",
            "torch-minigrid-history-learnability",
            "--invariant-history",
            "--qualification-method",
            method,
            "--acquisition-only",
            "--distinct-probe",
            "--old-task-readouts",
            "--return-to-a",
            "--train-a-rollout",
            *(["--describe"] if describe else []),
        ],
    )
    cli.main()
    assert calls == [
        dict(
            public_history=True,
            invariant_history=True,
            qualification_method=method,
            acquisition_only=True,
            distinct_probe=True,
            old_task_readouts=True,
            return_to_a=True,
            train_a_rollout=True,
        )
    ]
    assert json.loads(capsys.readouterr().out)["status"] == "complete"


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--scenario", "torch-minigrid-learnability"],
        ["--scenario", "torch-minigrid-history-learnability", "--invariant-history"],
        [
            "--scenario",
            "torch-minigrid-history-learnability",
            "--invariant-history",
            "--qualification-method",
            "bp",
        ],
        [
            "--scenario",
            "torch-minigrid-history-learnability",
            "--invariant-history",
            "--pc-qualification",
            "--acquisition-only",
        ],
    ],
)
def test_should_refuse_unfrozen_live_routes_before_dispatch(monkeypatch, arguments):
    monkeypatch.setattr(cli, "describe_minigrid_learnability_protocol", forbidden)
    monkeypatch.setattr(cli, "run_minigrid_learnability_pilot", forbidden)
    monkeypatch.setattr(sys, "argv", ["pilot", *arguments, "--train-a-rollout", "--describe"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
