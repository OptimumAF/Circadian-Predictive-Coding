"""A return is explicit and carries its existing isolated-probe prerequisites."""

import json
import sys
from typing import Any

import pytest

import scripts.run_coupled_attribution as cli


def forbidden(*args, **kwargs):
    raise AssertionError("invalid return scope reached resources")


@pytest.mark.parametrize("method", [None, "bp", "pc", "neutral_cpc", "gated_cpc"])
@pytest.mark.parametrize("describe", [False, True])
def test_should_dispatch_return_with_fixed_existing_method(monkeypatch, capsys, method, describe):
    calls = []

    def selected(**options):
        calls.append(options)
        return dict(status="complete")

    monkeypatch.setattr(
        cli, "describe_minigrid_learnability_protocol", selected if describe else forbidden
    )
    monkeypatch.setattr(cli, "run_minigrid_learnability_pilot", forbidden if describe else selected)
    args = [
        "pilot",
        "--scenario",
        "torch-minigrid-history-learnability",
        "--invariant-history",
        "--distinct-probe",
        "--old-task-readouts",
        "--return-to-a",
    ]
    expected: dict[str, Any] = dict(
        public_history=True,
        invariant_history=True,
        distinct_probe=True,
        old_task_readouts=True,
        return_to_a=True,
    )
    if method is not None:
        args += ["--qualification-method", method]
        expected["qualification_method"] = method
    if describe:
        args.append("--describe")
    monkeypatch.setattr(sys, "argv", args)
    cli.main()
    assert calls == [expected] and json.loads(capsys.readouterr().out)["status"] == "complete"


@pytest.mark.parametrize(
    "options",
    [
        [],
        ["--old-task-readouts"],
        [
            "--scenario",
            "torch-minigrid-history-learnability",
            "--invariant-history",
            "--distinct-probe",
        ],
        ["--scenario", "torch-minigrid-learnability", "--old-task-readouts"],
        ["--scenario", "torch-minigrid-history-learnability", "--old-task-readouts"],
    ],
)
def test_should_refuse_incomplete_return_scope_before_resources(monkeypatch, options):
    for name in ("describe_minigrid_learnability_protocol", "run_minigrid_learnability_pilot"):
        monkeypatch.setattr(cli, name, forbidden)
    monkeypatch.setattr(sys, "argv", ["pilot", "--return-to-a", "--describe", *options])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
