"""Public BP update observer admission and dispatch without native learner work."""

import json
import sys

import pytest

import scripts.run_coupled_attribution as cli
from test_minigrid_joint_catalogue_route import CLI_FLAGS, FLAGS
from test_torch_feature_rehearsal_pilot import forbidden


def arguments(method="bp"):
    selected = [*CLI_FLAGS, "--joint-catalogue"]
    selected[selected.index("--qualification-method") + 1] = method
    return selected


@pytest.mark.parametrize("method", ["bp"])
@pytest.mark.parametrize("describe", [False, True])
def test_should_dispatch_bp_observer_on_existing_selected_route(
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
    flags = arguments(method)
    if describe:
        flags.append("--describe")
    monkeypatch.setattr(sys, "argv", ["pilot", *flags, "--bp-update-diagnostics"])
    cli.main()
    assert calls == [
        {
            **FLAGS,
            "qualification_method": method,
            "joint_catalogue": True,
            "bp_update_diagnostics": True,
        }
    ]
    assert json.loads(capsys.readouterr().out)["status"] == "complete"


@pytest.mark.parametrize("describe", [False, True])
def test_should_omit_disabled_bp_observer_from_original_dispatch(monkeypatch, capsys, describe):
    calls = []

    def selected(**options):
        calls.append(options)
        return dict(status="complete")

    monkeypatch.setattr(
        cli, "describe_minigrid_learnability_protocol", selected if describe else forbidden
    )
    monkeypatch.setattr(cli, "run_minigrid_learnability_pilot", forbidden if describe else selected)
    flags = arguments()
    if describe:
        flags.append("--describe")
    monkeypatch.setattr(sys, "argv", ["pilot", *flags])
    cli.main()
    assert calls == [{**FLAGS, "joint_catalogue": True}]
    assert json.loads(capsys.readouterr().out)["status"] == "complete"


def refuse(monkeypatch, flags):
    errors = []
    original_error = cli.argparse.ArgumentParser.error

    def invalid(parser, message):
        errors.append(message)
        return original_error(parser, message)

    monkeypatch.setattr(cli.argparse.ArgumentParser, "error", invalid)
    monkeypatch.setattr(cli, "describe_minigrid_learnability_protocol", forbidden)
    monkeypatch.setattr(cli, "run_minigrid_learnability_pilot", forbidden)
    monkeypatch.setattr(sys, "argv", ["pilot", *flags, "--bp-update-diagnostics", "--describe"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert len(errors) == 1 and errors[0].startswith("--bp-update-diagnostics requires")


@pytest.mark.parametrize(
    "missing",
    [
        "--joint-catalogue",
        "--distinct-probe",
        "--old-task-readouts",
        "--return-to-a",
        "--invariant-history",
        "--qualification-method",
    ],
)
def test_should_refuse_partial_bp_observer_recipe_before_dispatch(monkeypatch, missing):
    flags = arguments()
    if missing == "--qualification-method":
        flags.remove("bp")
    flags.remove(missing)
    refuse(monkeypatch, flags)


@pytest.mark.parametrize(
    "method", ["pc", "neutral_cpc", "gated_cpc", "pc_relax64", "neutral_cpc_relax64"]
)
def test_should_refuse_different_native_recipes_before_dispatch(monkeypatch, method):
    refuse(monkeypatch, arguments(method))


def test_should_refuse_implicit_pc_alias_before_dispatch(monkeypatch):
    flags = arguments()
    flags.remove("--qualification-method")
    flags.remove("bp")
    refuse(monkeypatch, [*flags, "--pc-qualification"])


@pytest.mark.parametrize(
    "extra",
    [
        ["--cue-relative-history"],
        ["--old-a-rehearsal"],
        ["--probe-inference", "fixed-weighted"],
        ["--probe-inference", "bounded-weighted"],
        ["--weighted-bp"],
        ["--weighted-pc"],
        ["--weighted-catalogue"],
        ["--acquisition-only"],
        ["--memory-control"],
        ["--native-activation-health"],
        ["--cue-gradient-geometry"],
        ["--prior-gradient-geometry"],
        ["--train-action-margins"],
        ["--pc-update-diagnostics"],
        ["--train-a-rollout"],
        ["--probe-rollouts"],
        ["--deferred-probe-rollouts"],
        ["--probe-decision-diagnostics"],
        ["--invariant-memory"],
        ["--invariant-fifo-memory"],
        ["--environment-seed", "1"],
        ["--heldout-environment-seed", "2"],
        ["--source-artifact", "unused.json"],
    ],
)
def test_should_refuse_incompatible_options_before_dispatch(monkeypatch, extra):
    refuse(monkeypatch, [*arguments(), *extra])


@pytest.mark.parametrize(
    "scenario",
    ["original", "torch-minigrid-learnability", "torch-trajectory-rehearsal"],
)
def test_should_refuse_other_scenarios_before_dispatch(monkeypatch, scenario):
    monkeypatch.setattr(cli, "describe_protocol", forbidden)
    monkeypatch.setattr(cli, "describe_trajectory_protocol", forbidden)
    flags = arguments()
    flags[flags.index("--scenario") + 1] = scenario
    refuse(monkeypatch, flags)


def test_should_allow_pure_train_fit_bounds_with_bp_observer(monkeypatch, capsys):
    calls = []

    def selected(**options):
        calls.append(options)
        return dict(status="complete")

    monkeypatch.setattr(cli, "describe_minigrid_learnability_protocol", selected)
    monkeypatch.setattr(cli, "run_minigrid_learnability_pilot", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        ["pilot", *arguments(), "--bp-update-diagnostics", "--train-fit-bounds", "--describe"],
    )
    cli.main()
    assert calls == [
        {**FLAGS, "joint_catalogue": True, "bp_update_diagnostics": True, "train_fit_bounds": True}
    ]
    assert json.loads(capsys.readouterr().out)["status"] == "complete"
