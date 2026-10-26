"""Explicit H32 deferred-route admission and previews without runtime resources."""

import json
import sys

import numpy as np
import pytest

import scripts.run_coupled_attribution as cli
from src.adapters import minigrid_deferred_probe_rollout as deferred
from src.adapters import minigrid_learnability as pilot
from test_minigrid_joint_catalogue_route import CLI_FLAGS, FLAGS, METHODS


WIDE_FLAGS = {**FLAGS, "joint_catalogue": True, "wide_neural": True}
DEFERRED_FLAGS = {**WIDE_FLAGS, "deferred_probe_rollouts": True}


def forbidden(*args, **kwargs):
    raise AssertionError("admission and describe must precede every runtime resource")


@pytest.fixture
def no_resources(monkeypatch):
    for name in ("require_torch", "ProcessRssSampler", "MiniGridKeyDoor", "_history_learner"):
        monkeypatch.setattr(pilot, name, forbidden)
    monkeypatch.setattr(pilot.shared, "_train_arm", forbidden)
    monkeypatch.setattr(pilot.shared, "_collect_sources", forbidden)
    monkeypatch.setattr(pilot, "_collect_probe_sources", forbidden)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("extended", [False, True])
def test_should_describe_unchanged_learning_and_explicit_post_training_h32_games_without_resources(
    no_resources, method, extended
):
    options = {**WIDE_FLAGS, "qualification_method": method}
    original = pilot.describe_protocol(**options)
    selected = pilot.describe_protocol(
        **options, deferred_probe_rollouts=True, extended_wide_probe_budget=extended
    )
    assert selected["protocol_id"] != original["protocol_id"]
    assert selected["protocol_id"].endswith("_deferred_probe_rollouts_v1")
    assert "probe_rollout" not in selected
    assert selected["methods"] == original["methods"] == [f"public_invariant_{method}"]
    for field in ("seed", "hidden_dim", "learning_rate", "momentum", "checkpoint_updates"):
        assert selected[field] == original[field]
    assert selected["hidden_dim"] == 32 and selected["seed"] == 17
    games = selected["deferred_probe_rollouts"]
    assert games["all_training_before_games"] is True
    assert games["source_updates"] == [0, 128]
    assert games["native_endpoint_updates"] == [16, 32, 144, 160]
    assert games["native_endpoint_training_rows"] == [144, 288, 7056, 7200]
    assert games["frozen_readers"] == 2 and games["frozen_restores"] == 4
    assert games["maximum_state_bytes"] == 256 * 1024
    assert games["maximum_payload_bytes"] == 1024 * 1024
    assert games["maximum_logical_payload_bytes"] == 620544
    assert "differs" in games["boundary_scope"]
    work = selected["planned_work"]
    for field in ("native_train_calls", "training_rows", "score_forwards", "score_rows"):
        assert work[field] == original["planned_work"][field]
    assert work["native_train_calls"] == 192 and work["training_rows"] == 7488
    assert work["score_forwards"] == 94 and work["score_rows"] == 846
    assert work["learner_live_episodes"] == 16
    assert work["maximum_live_prediction_forwards"] == 256
    assert work["maximum_live_environment_steps"] == 256
    assert work["final_episodes"] == work["replay_rows"] == 0
    assert selected["primary_comparisons"] == []
    assert selected["limits"]["body_seconds"] == (240 if extended else 180)
    assert ("60s" if extended else "40s") in games["budget"]
    if extended:
        assert games["endpoint_wall_seconds"] == 60
        assert games["parent_wall_seconds"] == 240
    else:
        assert "endpoint_wall_seconds" not in games


@pytest.mark.parametrize("options", [{}, FLAGS, WIDE_FLAGS])
def test_should_preserve_disabled_deferred_descriptor_exactly(no_resources, options):
    original = pilot.describe_protocol(**options)
    assert pilot.describe_protocol(**options, deferred_probe_rollouts=False) == original
    assert "deferred_probe_rollouts" not in original


@pytest.mark.parametrize(
    "changed",
    [
        {"public_history": False},
        {"invariant_history": False},
        {"joint_catalogue": False},
        {"wide_neural": False},
        {"qualification_method": None},
        {"qualification_method": "recurrent_bc"},
        {"qualification_method": "pc_relax64"},
        {"qualification_method": np.str_("bp")},
        {"distinct_probe": False},
        {"old_task_readouts": False},
        {"return_to_a": False},
        {"memory_control": True},
        {"probe_rollouts": True},
        {"activation_health": True},
        {"gradient_geometry": True},
        {"prior_gradient_geometry": True},
        {"acquisition_only": True},
        {"weighted_bp": True},
        {"old_a_rehearsal": True},
        {"train_a_rollout": True},
        {"probe_inference": "fixed-weighted"},
    ],
)
def test_should_refuse_incomplete_or_conflicting_deferred_h32_recipe_before_resources(
    no_resources, changed
):
    for operation in (pilot.describe_protocol, pilot.run_pilot):
        with pytest.raises(ValueError):
            operation(**{**DEFERRED_FLAGS, **changed})


@pytest.mark.parametrize("flag", [None, 0, 1, "true", np.bool_(True)])
def test_should_refuse_non_native_deferred_flag_before_resources(no_resources, flag):
    for operation in (pilot.describe_protocol, pilot.run_pilot):
        with pytest.raises(ValueError):
            operation(**{**DEFERRED_FLAGS, "deferred_probe_rollouts": flag})


@pytest.mark.parametrize("method", METHODS)
def test_should_admit_only_explicit_neural_or_legacy_count_recipe(method):
    deferred.validate_option(True, False, False, wide_method=method)
    deferred.validate_option(True, True, False)
    for memory, inline in ((True, False), (False, True), (0, False), (False, 0)):
        with pytest.raises(ValueError):
            deferred.validate_option(True, memory, inline, wide_method=method)


@pytest.mark.parametrize("describe", [False, True])
@pytest.mark.parametrize("extended", [False, True])
def test_should_forward_both_explicit_deferred_h32_and_resource_selection(
    monkeypatch, capsys, describe, extended
):
    calls = []

    def selected(**options):
        calls.append(options)
        return {"status": "complete"}

    monkeypatch.setattr(
        cli, "describe_minigrid_learnability_protocol", selected if describe else forbidden
    )
    monkeypatch.setattr(cli, "run_minigrid_learnability_pilot", forbidden if describe else selected)
    monkeypatch.setattr(
        sys,
        "argv",
        ["pilot", *CLI_FLAGS, "--joint-catalogue", "--wide-neural", "--deferred-probe-rollouts"]
        + (["--extended-wide-probe-budget"] if extended else [])
        + (["--describe"] if describe else []),
    )
    cli.main()
    assert calls == [
        {**DEFERRED_FLAGS, **({"extended_wide_probe_budget": True} if extended else {})}
    ]
    assert json.loads(capsys.readouterr().out) == {"status": "complete"}


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        CLI_FLAGS,
        [*CLI_FLAGS, "--wide-neural"],
        [*CLI_FLAGS, "--joint-catalogue"],
        [*CLI_FLAGS, "--joint-catalogue", "--wide-neural", "--probe-rollouts"],
        [*CLI_FLAGS, "--joint-catalogue", "--wide-neural", "--memory-control"],
        [*CLI_FLAGS, "--joint-catalogue", "--wide-neural", "--train-a-rollout"],
        [*CLI_FLAGS, "--joint-catalogue", "--wide-neural", "--old-a-rehearsal"],
    ],
)
def test_should_refuse_wrong_cli_deferred_routes_before_any_dispatch(monkeypatch, arguments):
    for name in (
        "describe_minigrid_learnability_protocol",
        "run_minigrid_learnability_pilot",
        "run_minigrid_pilot",
    ):
        monkeypatch.setattr(cli, name, forbidden)
    monkeypatch.setattr(
        sys, "argv", ["pilot", *arguments, "--deferred-probe-rollouts", "--describe"]
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_should_keep_incomplete_native_game_report_and_nonzero_cli_exit(monkeypatch, capsys):
    incomplete = dict(
        status="incomplete",
        reason="deferred native endpoint deadline",
        deferred_probe_rollouts=dict(status="failed", records=[]),
    )
    monkeypatch.setattr(cli, "run_minigrid_learnability_pilot", lambda **options: incomplete)
    monkeypatch.setattr(
        sys,
        "argv",
        ["pilot", *CLI_FLAGS, "--joint-catalogue", "--wide-neural", "--deferred-probe-rollouts"],
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert json.loads(capsys.readouterr().out) == incomplete
