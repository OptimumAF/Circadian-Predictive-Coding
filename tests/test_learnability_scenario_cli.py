"""Fixed-scenario CLI routing with stubs; no native pilot execution."""

import json
import sys

import pytest

import scripts.run_learnability_pilot as cli


def forbidden(*args, **kwargs):
    pytest.fail("an unselected operational route was reached")


@pytest.mark.parametrize("action", ("describe", "run"))
@pytest.mark.parametrize("scenario", ("four-update", "sixteen-update"))
def test_should_select_exactly_the_declared_action_and_fixed_scenario(
    monkeypatch, capsys, action, scenario
):
    calls = []

    def selected(**kwargs):
        calls.append(kwargs)
        return {"status": "complete", "scenario": scenario}

    monkeypatch.setattr(cli, "describe_protocol", forbidden)
    monkeypatch.setattr(cli, "run_pilot", forbidden)
    monkeypatch.setattr(cli, "describe_protocol" if action == "describe" else "run_pilot", selected)
    monkeypatch.setattr(sys, "argv", ["pilot", f"--{action}", "--scenario", scenario])
    assert cli.main() == 0
    assert calls == ([{"extended": True}] if scenario == "sixteen-update" else [{}])
    assert json.loads(capsys.readouterr().out) == {"status": "complete", "scenario": scenario}


@pytest.mark.parametrize("scenario", ("sixteen", "adaptive"))
def test_should_refuse_unregistered_scenarios_before_any_callback(monkeypatch, scenario):
    monkeypatch.setattr(cli, "describe_protocol", forbidden)
    monkeypatch.setattr(cli, "run_pilot", forbidden)
    monkeypatch.setattr(sys, "argv", ["pilot", "--run", "--scenario", scenario])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


@pytest.mark.parametrize(
    "option,value", (("--seed", "41"), ("--probe-updates", "8"), ("--learning-rate", "0.1"))
)
def test_should_refuse_unfrozen_experiment_controls_before_any_callback(monkeypatch, option, value):
    monkeypatch.setattr(cli, "describe_protocol", forbidden)
    monkeypatch.setattr(cli, "run_pilot", forbidden)
    monkeypatch.setattr(
        sys, "argv", ["pilot", "--run", "--scenario", "sixteen-update", option, value]
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


@pytest.mark.parametrize("scenario", ("four-update", "sixteen-update"))
def test_should_emit_incomplete_output_and_return_failure_without_retry(
    monkeypatch, capsys, scenario
):
    calls = []

    def selected(**kwargs):
        calls.append(kwargs)
        return {"status": "incomplete", "reason": "retained failure", "runs": []}

    monkeypatch.setattr(cli, "describe_protocol", forbidden)
    monkeypatch.setattr(cli, "run_pilot", selected)
    monkeypatch.setattr(sys, "argv", ["pilot", "--run", "--scenario", scenario])
    assert cli.main() == 2
    assert len(calls) == 1
    assert json.loads(capsys.readouterr().out)["reason"] == "retained failure"
