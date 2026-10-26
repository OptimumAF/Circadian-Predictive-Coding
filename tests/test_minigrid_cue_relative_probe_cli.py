"""Existing public CLI dispatch for explicit cue-relative F/old-A/return mode."""

import json
import sys

import pytest

import scripts.run_coupled_attribution as cli
from test_minigrid_joint_catalogue_route import CLI_FLAGS, FLAGS
from test_torch_feature_rehearsal_pilot import forbidden


@pytest.mark.parametrize("method", ["bp", "pc", "neutral_cpc", "gated_cpc"])
@pytest.mark.parametrize("describe", [False, True])
def test_should_dispatch_cue_relative_on_existing_selected_route(
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
    flags = list(CLI_FLAGS)
    flags[flags.index("--qualification-method") + 1] = method
    arguments = ["pilot", *flags, "--joint-catalogue", "--cue-relative-history"]
    if describe:
        arguments.append("--describe")
    monkeypatch.setattr(sys, "argv", arguments)
    cli.main()
    assert calls == [
        {
            **FLAGS,
            "qualification_method": method,
            "joint_catalogue": True,
            "cue_relative_history": True,
        }
    ]
    assert json.loads(capsys.readouterr().out)["status"] == "complete"


@pytest.mark.parametrize(
    "missing",
    [
        "--joint-catalogue",
        "--distinct-probe",
        "--old-task-readouts",
        "--return-to-a",
        "--invariant-history",
    ],
)
def test_should_refuse_partial_route_before_dispatch(monkeypatch, missing):
    monkeypatch.setattr(cli, "describe_minigrid_learnability_protocol", forbidden)
    monkeypatch.setattr(cli, "run_minigrid_learnability_pilot", forbidden)
    arguments = [*CLI_FLAGS, "--joint-catalogue"]
    arguments.remove(missing)
    monkeypatch.setattr(sys, "argv", ["pilot", *arguments, "--cue-relative-history", "--describe"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
