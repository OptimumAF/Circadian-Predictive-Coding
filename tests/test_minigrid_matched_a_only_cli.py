"""Matched A/F/A recipes use the existing explicit native-method CLI selector."""

import json
import sys

import pytest

import scripts.run_coupled_attribution as cli
from test_torch_feature_rehearsal_pilot import forbidden


@pytest.mark.parametrize("method", ["bp", "pc", "neutral_cpc", "gated_cpc"])
@pytest.mark.parametrize("describe", [True, False])
def test_should_dispatch_matched_a_f_a_recipe_without_changing_method(
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
        )
    ]
    assert json.loads(capsys.readouterr().out)["status"] == "complete"
