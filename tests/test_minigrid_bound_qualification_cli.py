"""Explicit existing-method singleton dispatch preserves legacy CLI routes."""

import json
import sys

import pytest

import scripts.run_coupled_attribution as cli
from test_torch_feature_rehearsal_pilot import forbidden


@pytest.mark.parametrize("method", ["bp", "pc", "neutral_cpc", "gated_cpc"])
@pytest.mark.parametrize("describe", [False, True])
@pytest.mark.parametrize("health", [False, True])
def test_should_dispatch_selected_native_family_without_other_work(
    monkeypatch, capsys, method, describe, health
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
        ]
        + (["--describe"] if describe else [])
        + (["--native-activation-health"] if health else []),
    )
    cli.main()
    expected = dict(public_history=True, invariant_history=True, qualification_method=method)
    if health:
        expected["activation_health"] = True
    assert calls == [expected]
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
        [
            "--scenario",
            "torch-minigrid-history-learnability",
            "--invariant-history",
            "--pc-qualification",
        ],
    ],
)
def test_should_refuse_wrong_routes_or_alias_conflict_before_work(monkeypatch, arguments):
    for name in (
        "describe_minigrid_learnability_protocol",
        "run_minigrid_learnability_pilot",
        "run_minigrid_pilot",
    ):
        monkeypatch.setattr(cli, name, forbidden)
    monkeypatch.setattr(
        sys, "argv", ["pilot", "--describe", "--qualification-method", "bp", *arguments]
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


@pytest.mark.parametrize("method", ["recurrent_bc", "public_invariant_bp", "active_cpc", "BP", ""])
def test_should_refuse_unsupported_method_without_resource_calls(monkeypatch, method):
    monkeypatch.setattr(cli, "describe_minigrid_learnability_protocol", forbidden)
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
            "--describe",
        ],
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_should_refuse_duplicate_pc_spelling_even_when_it_names_the_same_method(monkeypatch):
    monkeypatch.setattr(cli, "describe_minigrid_learnability_protocol", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pilot",
            "--scenario",
            "torch-minigrid-history-learnability",
            "--invariant-history",
            "--qualification-method",
            "pc",
            "--pc-qualification",
            "--describe",
        ],
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
