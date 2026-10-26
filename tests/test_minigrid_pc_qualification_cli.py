"""Explicit one-control selection through the existing CLI only."""

import json
import sys

import pytest

import scripts.run_coupled_attribution as cli
from test_torch_feature_rehearsal_pilot import forbidden


@pytest.mark.parametrize("describe", [False, True])
def test_should_dispatch_exact_qualification_flags_without_running_other_scenarios(
    monkeypatch, capsys, describe
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
            "--pc-qualification",
        ]
        + (["--describe"] if describe else []),
    )
    cli.main()
    assert calls == [dict(public_history=True, invariant_history=True, pc_qualification=True)]
    assert json.loads(capsys.readouterr().out)["status"] == "complete"


@pytest.mark.parametrize(
    "arguments",
    [
        ["--scenario", "torch-minigrid-history-learnability"],
        ["--scenario", "torch-minigrid-learnability", "--invariant-history"],
        ["--scenario", "torch-minigrid-cue-key-door", "--invariant-history"],
        [
            "--scenario",
            "torch-minigrid-history-learnability",
            "--invariant-history",
            "--cue-gradient-geometry",
        ],
    ],
)
def test_should_refuse_unfrozen_qualification_routes_before_selected_work(monkeypatch, arguments):
    for name in (
        "describe_minigrid_learnability_protocol",
        "run_minigrid_learnability_pilot",
        "run_minigrid_pilot",
    ):
        monkeypatch.setattr(cli, name, forbidden)
    monkeypatch.setattr(sys, "argv", ["pilot", "--describe", "--pc-qualification", *arguments])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
