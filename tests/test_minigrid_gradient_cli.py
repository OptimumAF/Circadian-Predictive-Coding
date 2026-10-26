"""CLI gradient opt-in stays explicit and confined to TRAIN learnability routes."""

import json
import sys

import pytest

import scripts.run_coupled_attribution as cli


@pytest.mark.parametrize("describe", [False, True])
@pytest.mark.parametrize("history,health", [(False, False), (True, False), (True, True)])
def test_should_forward_explicit_gradient_option_to_only_the_chosen_route(
    monkeypatch, capsys, describe, history, health
):
    calls = []

    def selected(**options):
        calls.append(options)
        return {"status": "complete", "selected": options}

    def forbidden(*args, **kwargs):
        pytest.fail("the unselected execution/describe route must not run")

    monkeypatch.setattr(
        cli, "describe_minigrid_learnability_protocol", selected if describe else forbidden
    )
    monkeypatch.setattr(cli, "run_minigrid_learnability_pilot", forbidden if describe else selected)
    scenario = "torch-minigrid-history-learnability" if history else "torch-minigrid-learnability"
    argv = ["pilot", "--scenario", scenario, "--cue-gradient-geometry"]
    if describe:
        argv.append("--describe")
    if health:
        argv.append("--native-activation-health")
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    expected = {"gradient_geometry": True}
    if history:
        expected["public_history"] = True
    if health:
        expected["activation_health"] = True
    assert calls == [expected]
    assert json.loads(capsys.readouterr().out) == {"status": "complete", "selected": expected}


@pytest.mark.parametrize(
    "scenario",
    ["torch-minigrid-cue-key-door", "torch-key-door-learnability", "torch-trajectory-rollouts"],
)
def test_should_refuse_other_scenarios_before_native_work(monkeypatch, capsys, scenario):
    monkeypatch.setattr(
        sys, "argv", ["pilot", "--scenario", scenario, "--cue-gradient-geometry", "--describe"]
    )
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2
    assert "requires a MiniGrid learnability scenario" in capsys.readouterr().err


@pytest.mark.parametrize("history", [False, True])
def test_should_keep_original_no_flag_keyword_contract(monkeypatch, capsys, history):
    calls = []

    def selected(**options):
        calls.append(options)
        return {"status": "complete", "selected": options}

    monkeypatch.setattr(cli, "describe_minigrid_learnability_protocol", selected)
    scenario = "torch-minigrid-history-learnability" if history else "torch-minigrid-learnability"
    monkeypatch.setattr(sys, "argv", ["pilot", "--scenario", scenario, "--describe"])
    cli.main()
    expected = {"public_history": True} if history else {}
    assert calls == [expected]
    assert json.loads(capsys.readouterr().out) == {"status": "complete", "selected": expected}
