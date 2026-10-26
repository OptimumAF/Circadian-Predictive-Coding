"""Memory control is a separate explicit option; neural selectors stay unchanged."""

import json
import sys

import pytest

import scripts.run_coupled_attribution as cli
from test_torch_feature_rehearsal_pilot import forbidden

FLAGS = [
    "--scenario",
    "torch-minigrid-history-learnability",
    "--invariant-history",
    "--acquisition-only",
    "--distinct-probe",
    "--old-task-readouts",
    "--return-to-a",
]


@pytest.mark.parametrize("describe", [True, False])
def test_should_dispatch_fixed_count_control_without_a_neural_selector(
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
        sys, "argv", ["pilot", *FLAGS, "--memory-control", *(["--describe"] if describe else [])]
    )
    cli.main()
    assert calls == [
        dict(
            public_history=True,
            invariant_history=True,
            acquisition_only=True,
            distinct_probe=True,
            old_task_readouts=True,
            return_to_a=True,
            memory_control=True,
        )
    ]
    assert json.loads(capsys.readouterr().out)["status"] == "complete"


@pytest.mark.parametrize(
    "missing",
    [
        "--invariant-history",
        "--acquisition-only",
        "--distinct-probe",
        "--old-task-readouts",
        "--return-to-a",
    ],
)
def test_should_refuse_incomplete_memory_recipe_before_dispatch(monkeypatch, missing):
    monkeypatch.setattr(cli, "describe_minigrid_learnability_protocol", forbidden)
    monkeypatch.setattr(cli, "run_minigrid_learnability_pilot", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        ["pilot", *(flag for flag in FLAGS if flag != missing), "--memory-control", "--describe"],
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


@pytest.mark.parametrize(
    "extra",
    [["--qualification-method", method] for method in ("bp", "pc", "neutral_cpc", "gated_cpc")]
    + [
        [flag]
        for flag in (
            "--pc-qualification",
            "--native-activation-health",
            "--cue-gradient-geometry",
            "--train-a-rollout",
        )
    ],
)
def test_should_refuse_neural_or_live_options_in_the_count_recipe(monkeypatch, extra):
    monkeypatch.setattr(cli, "describe_minigrid_learnability_protocol", forbidden)
    monkeypatch.setattr(cli, "run_minigrid_learnability_pilot", forbidden)
    monkeypatch.setattr(sys, "argv", ["pilot", *FLAGS, "--memory-control", *extra, "--describe"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
