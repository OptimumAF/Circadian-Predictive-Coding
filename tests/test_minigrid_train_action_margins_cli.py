"""Explicit TRAIN margins preserve the frozen route and refuse early misuse."""

import json
import sys

import numpy as np
import pytest

import scripts.run_coupled_attribution as cli
from src.adapters import minigrid_learnability as pilot


FLAGS = dict(public_history=True, invariant_history=True)
CLI_FLAGS = ["--scenario", "torch-minigrid-history-learnability", "--invariant-history"]


def forbidden(*args, **kwargs):
    pytest.fail("preflight reached an operational resource or unselected callback")


@pytest.mark.parametrize("options", [{}, FLAGS, dict(public_history=True)])
def test_should_preserve_default_protocol_when_margins_are_false(options):
    assert pilot.describe_protocol(**options) == pilot.describe_protocol(
        **options, train_action_margins=False
    )


@pytest.mark.parametrize("method", [None, "bp", "pc", "neutral_cpc", "gated_cpc"])
def test_should_describe_all_paid_train_rows_without_new_native_work(monkeypatch, method):
    for name in ("require_torch", "ProcessRssSampler", "MiniGridKeyDoor", "_history_learner"):
        monkeypatch.setattr(pilot, name, forbidden)
    options = {**FLAGS, **({"qualification_method": method} if method is not None else {})}
    ordinary = pilot.describe_protocol(**options)
    protocol = pilot.describe_protocol(**options, train_action_margins=True)

    description = protocol.pop("train_action_margins")
    planned = protocol["planned_work"].pop("train_action_margins")
    assert protocol == ordinary
    assert description["definition"] == "categorical_target_probability_margin_v1"
    assert planned == dict(
        reductions=ordinary["planned_work"]["known_raw_score_forwards"],
        rows=ordinary["planned_work"]["known_raw_score_rows"],
        input_array_bytes=ordinary["planned_work"]["known_raw_score_rows"] * 8 * 8,
        additional_native_forwards=0,
        optimizer_steps=0,
        published_updates=0,
    )


@pytest.mark.parametrize(
    "options",
    [
        dict(train_action_margins=True),
        dict(train_action_margins=True, public_history=True),
        dict(train_action_margins=True, invariant_history=True),
        dict(train_action_margins=1, **FLAGS),
        dict(train_action_margins=0, **FLAGS),
        dict(train_action_margins=None, **FLAGS),
        dict(train_action_margins=np.bool_(True), **FLAGS),
        dict(train_action_margins=np.bool_(False), **FLAGS),
    ],
)
def test_should_refuse_unsupported_api_options_before_resource_construction(monkeypatch, options):
    for name in ("require_torch", "ProcessRssSampler", "MiniGridKeyDoor", "_history_learner"):
        monkeypatch.setattr(pilot, name, forbidden)
    with pytest.raises(ValueError):
        pilot.describe_protocol(**options)
    with pytest.raises(ValueError):
        pilot.run_pilot(**options)


@pytest.mark.parametrize("describe", [False, True])
@pytest.mark.parametrize("margins", [False, True])
def test_should_forward_cli_margins_only_when_explicit(monkeypatch, capsys, describe, margins):
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
        ["pilot", *CLI_FLAGS]
        + (["--train-action-margins"] if margins else [])
        + (["--describe"] if describe else []),
    )
    cli.main()
    assert calls == [{**FLAGS, **({"train_action_margins": True} if margins else {})}]
    assert json.loads(capsys.readouterr().out)["status"] == "complete"


@pytest.mark.parametrize("describe", [False, True])
@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--scenario", "torch-minigrid-learnability"],
        ["--scenario", "torch-minigrid-cue-key-door"],
        ["--scenario", "torch-minigrid-history-learnability"],
    ],
)
def test_should_refuse_cli_margins_without_the_supported_history_route(
    monkeypatch, describe, arguments
):
    for name in (
        "describe_minigrid_learnability_protocol",
        "run_minigrid_learnability_pilot",
        "run_minigrid_pilot",
    ):
        monkeypatch.setattr(cli, name, forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        ["pilot", *arguments, "--train-action-margins"] + (["--describe"] if describe else []),
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
