"""A-only acquisition is an explicit existing native qualification recipe."""

import json
import sys

import pytest

import scripts.run_coupled_attribution as cli
from test_torch_feature_rehearsal_pilot import forbidden


@pytest.mark.parametrize("method", ["bp", "pc", "neutral_cpc", "gated_cpc"])
@pytest.mark.parametrize("describe", [False, True])
def test_should_dispatch_exact_a_only_recipe_without_extra_calls(
    monkeypatch, capsys, describe, method
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
        ]
        + (["--describe"] if describe else []),
    )
    cli.main()
    assert calls == [
        dict(
            public_history=True,
            invariant_history=True,
            qualification_method=method,
            acquisition_only=True,
        )
    ]
    assert json.loads(capsys.readouterr().out)["status"] == "complete"


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        [
            "--scenario",
            "torch-minigrid-learnability",
            "--invariant-history",
            "--qualification-method",
            "bp",
        ],
        [
            "--scenario",
            "torch-minigrid-cue-key-door",
            "--invariant-history",
            "--qualification-method",
            "bp",
        ],
        ["--scenario", "torch-minigrid-history-learnability", "--qualification-method", "bp"],
        ["--scenario", "torch-minigrid-history-learnability", "--invariant-history"],
        [
            "--scenario",
            "torch-minigrid-history-learnability",
            "--invariant-history",
            "--pc-qualification",
        ],
        *[
            [
                "--scenario",
                "torch-minigrid-history-learnability",
                "--invariant-history",
                "--qualification-method",
                method,
            ]
            for method in ("recurrent_bc", "unsupported")
        ],
        *[
            [
                "--scenario",
                "torch-minigrid-history-learnability",
                "--invariant-history",
                "--qualification-method",
                "bp",
                flag,
            ]
            for flag in ("--native-activation-health", "--cue-gradient-geometry")
        ],
    ],
)
def test_should_refuse_unfrozen_a_only_combinations_before_resources(monkeypatch, arguments):
    for name in (
        "describe_minigrid_learnability_protocol",
        "run_minigrid_learnability_pilot",
        "run_minigrid_pilot",
    ):
        monkeypatch.setattr(cli, name, forbidden)
    monkeypatch.setattr(sys, "argv", ["pilot", "--describe", "--acquisition-only", *arguments])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
