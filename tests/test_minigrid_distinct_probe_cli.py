"""Distinct task is an opt-in on the existing bound learnability route."""

import json
import sys
from typing import Any
import pytest
import scripts.run_coupled_attribution as cli


def forbidden(*args, **kwargs):
    raise AssertionError("unsupported route reached resources")


@pytest.mark.parametrize("method", [None, "bp", "pc", "neutral_cpc", "gated_cpc"])
@pytest.mark.parametrize("describe", [False, True])
def test_should_dispatch_new_probe_without_altering_selected_native_method(
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
    arguments = [
        "pilot",
        "--scenario",
        "torch-minigrid-history-learnability",
        "--invariant-history",
        "--distinct-probe",
    ]
    expected: dict[str, Any] = dict(
        public_history=True, invariant_history=True, distinct_probe=True
    )
    if method is not None:
        arguments += ["--qualification-method", method]
        expected["qualification_method"] = method
    if describe:
        arguments += ["--describe"]
    monkeypatch.setattr(sys, "argv", arguments)
    cli.main()
    assert calls == [expected] and json.loads(capsys.readouterr().out)["status"] == "complete"


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--scenario", "torch-minigrid-learnability"],
        ["--scenario", "torch-minigrid-cue-key-door"],
        ["--scenario", "torch-minigrid-history-learnability"],
        ["--scenario", "torch-minigrid-history-learnability", "--qualification-method", "bp"],
    ],
)
def test_should_refuse_unsupported_probe_routes_before_resources(monkeypatch, arguments):
    for name in (
        "describe_minigrid_learnability_protocol",
        "run_minigrid_learnability_pilot",
        "run_minigrid_pilot",
    ):
        monkeypatch.setattr(cli, name, forbidden)
    monkeypatch.setattr(sys, "argv", ["pilot", "--describe", "--distinct-probe", *arguments])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
