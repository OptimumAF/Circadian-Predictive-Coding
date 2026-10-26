"""Old A fit is an explicit optional monitor of the distinct-task route."""

import json
import sys
from typing import Any
import pytest
import scripts.run_coupled_attribution as cli


def forbidden(*args, **kwargs):
    raise AssertionError("invalid route reached resources")


@pytest.mark.parametrize("method", [None, "bp", "pc", "neutral_cpc", "gated_cpc"])
@pytest.mark.parametrize("describe", [False, True])
def test_should_dispatch_fixed_old_fit_without_altering_the_selected_native_method(
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
        "--old-task-readouts",
    ]
    expected: dict[str, Any] = dict(
        public_history=True, invariant_history=True, distinct_probe=True, old_task_readouts=True
    )
    if method is not None:
        arguments += ["--qualification-method", method]
        expected["qualification_method"] = method
    if describe:
        arguments.append("--describe")
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
        ["--scenario", "torch-minigrid-history-learnability", "--invariant-history"],
        ["--scenario", "torch-minigrid-history-learnability", "--distinct-probe"],
    ],
)
def test_should_refuse_missing_scope_before_resources(monkeypatch, arguments):
    for name in (
        "describe_minigrid_learnability_protocol",
        "run_minigrid_learnability_pilot",
        "run_minigrid_pilot",
    ):
        monkeypatch.setattr(cli, name, forbidden)
    monkeypatch.setattr(sys, "argv", ["pilot", "--describe", "--old-task-readouts", *arguments])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
