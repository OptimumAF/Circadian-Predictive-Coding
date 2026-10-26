"""Explicit H32 old-A rehearsal admission and CLI dispatch without resources.

These checks construct no native learners and execute no training, prediction,
physics or comparative experiments. Existing H8/default routes stay available.
"""

import json
import sys

import numpy as np
import pytest

import scripts.run_coupled_attribution as cli
from src.adapters import minigrid_learnability as pilot
from src.adapters import minigrid_probe_rehearsal as recipe
from test_minigrid_joint_catalogue_route import CLI_FLAGS, FLAGS, METHODS


WIDE_FLAGS = {**FLAGS, "joint_catalogue": True, "wide_neural": True}
REHEARSAL_FLAGS = {**WIDE_FLAGS, "old_a_rehearsal": True}


def forbidden(*args, **kwargs):
    raise AssertionError("admission and describe must precede runtime resources")


@pytest.fixture
def no_resources(monkeypatch):
    for name in ("require_torch", "ProcessRssSampler", "MiniGridKeyDoor", "_history_learner"):
        monkeypatch.setattr(pilot, name, forbidden)
    monkeypatch.setattr(pilot.shared, "_train_arm", forbidden)
    monkeypatch.setattr(pilot.shared, "_collect_sources", forbidden)
    monkeypatch.setattr(pilot, "_collect_probe_sources", forbidden)


@pytest.mark.parametrize("method", METHODS)
def test_should_describe_extra_h32_training_without_changing_original_doses(no_resources, method):
    options = {**WIDE_FLAGS, "qualification_method": method}
    original = pilot.describe_protocol(**options)
    selected = pilot.describe_protocol(**options, old_a_rehearsal=True)
    assert selected["protocol_id"] != original["protocol_id"]
    assert "old_a_rehearsal" in selected["protocol_id"]
    assert selected["methods"] == original["methods"] == [f"public_invariant_{method}"]
    for field in ("seed", "hidden_dim", "learning_rate", "momentum", "checkpoint_updates"):
        assert selected[field] == original[field]
    assert selected["hidden_dim"] == 32 and selected["seed"] == 17
    assert selected["probe"]["source_updates"] == [0, 128]
    assert selected["probe"]["updates"] == selected["return_phase"]["updates"] == 16
    assert selected["primary_comparisons"] == original["primary_comparisons"] == []
    work = selected["planned_work"]
    assert work["benchmark_native_train_calls"] == 128
    assert work["benchmark_training_rows"] == 6912
    assert work["native_train_calls"] == 256 and work["training_rows"] == 8064
    assert work["replay_rows"] == 576
    assert work["score_forwards"] == 94 and work["score_rows"] == 846
    assert work["final_episodes"] == work["learner_live_episodes"] == 0
    native = selected["wide_neural"]
    assert native["bp_optimizer_steps_derived"] == (256 if method == "bp" else 0)
    assert native["pc_parameter_updates_derived"] == (0 if method == "bp" else 256)
    assert native["nominal_latent_row_steps"] == (0 if method == "bp" else 16128)
    assert native["replay_native_calls"] == 64 and native["replay_training_rows"] == 576
    replay = selected["old_a_rehearsal"]
    assert replay["native_calls_per_probe"] == 32 and replay["replay_rows_per_probe"] == 288
    assert replay["bank_transitions_per_probe"] == 18
    assert replay["bank_encoded_bytes_per_probe"] == 87280
    assert "no further replay" in replay["return_phase"]
    assert "not matched" in replay["comparison"]


@pytest.mark.parametrize("options", [{}, FLAGS, WIDE_FLAGS, {**FLAGS, "joint_catalogue": True}])
def test_should_preserve_disabled_rehearsal_descriptor_exactly(no_resources, options):
    assert pilot.describe_protocol(**options) == pilot.describe_protocol(
        **options, old_a_rehearsal=False
    )


@pytest.mark.parametrize("flag", [None, 0, 1, "true", np.bool_(True)])
@pytest.mark.parametrize("name", ["wide_neural", "old_a_rehearsal"])
def test_should_refuse_non_native_bool_before_resources(no_resources, flag, name):
    for operation in (pilot.describe_protocol, pilot.run_pilot):
        with pytest.raises(ValueError):
            operation(**{**REHEARSAL_FLAGS, name: flag})


@pytest.mark.parametrize(
    "changed",
    [
        {"public_history": False},
        {"invariant_history": False},
        {"joint_catalogue": False},
        {"distinct_probe": False},
        {"old_task_readouts": False},
        {"return_to_a": False},
        {"qualification_method": None},
        {"qualification_method": "pc_relax64"},
        {"qualification_method": "public_invariant_bp"},
        {"qualification_method": np.str_("bp")},
        {"memory_control": True},
        {"weighted_bp": True},
        {"weighted_pc": True},
        {"weighted_catalogue": True},
        {"probe_rollouts": True},
        {"deferred_probe_rollouts": True},
        {"extended_wide_probe_budget": True},
        {"train_a_rollout": True},
        {"activation_health": True},
        {"gradient_geometry": True},
        {"prior_gradient_geometry": True},
        {"pc_update_diagnostics": True},
        {"bp_update_diagnostics": True},
        {"probe_inference": "fixed-weighted"},
    ],
)
def test_should_refuse_incomplete_or_conflicting_h32_replay_before_resources(no_resources, changed):
    for operation in (pilot.describe_protocol, pilot.run_pilot):
        with pytest.raises(ValueError):
            operation(**{**REHEARSAL_FLAGS, **changed})


@pytest.mark.parametrize("value", [None, 0, 1, "true", np.bool_(True)])
def test_should_validate_explicit_width_at_pure_rehearsal_boundary(value):
    with pytest.raises(ValueError):
        recipe.validate_option(True, **{**WIDE_FLAGS, "wide_neural": value})


def arguments(method="bp"):
    flags = [*CLI_FLAGS, "--joint-catalogue", "--wide-neural"]
    flags[flags.index("--qualification-method") + 1] = method
    return flags


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("describe", [False, True])
def test_should_forward_explicit_h32_replay_on_existing_route(
    monkeypatch, capsys, method, describe
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
        ["pilot", *arguments(method), "--old-a-rehearsal"] + (["--describe"] if describe else []),
    )
    cli.main()
    assert calls == [{**REHEARSAL_FLAGS, "qualification_method": method}]
    assert json.loads(capsys.readouterr().out) == {"status": "complete"}


@pytest.mark.parametrize("describe", [False, True])
def test_should_keep_wide_no_replay_dispatch_unchanged(monkeypatch, capsys, describe):
    calls = []

    def selected(**options):
        calls.append(options)
        return {"status": "complete"}

    monkeypatch.setattr(
        cli, "describe_minigrid_learnability_protocol", selected if describe else forbidden
    )
    monkeypatch.setattr(cli, "run_minigrid_learnability_pilot", forbidden if describe else selected)
    monkeypatch.setattr(sys, "argv", ["pilot", *arguments()] + (["--describe"] if describe else []))
    cli.main()
    assert calls == [WIDE_FLAGS]
    assert json.loads(capsys.readouterr().out) == {"status": "complete"}


@pytest.mark.parametrize(
    "extra",
    [
        ["--probe-rollouts"],
        ["--deferred-probe-rollouts"],
        ["--extended-wide-probe-budget"],
        ["--memory-control"],
        ["--weighted-bp"],
        ["--train-a-rollout"],
        ["--native-activation-health"],
        ["--pc-update-diagnostics"],
        ["--probe-inference", "fixed-weighted"],
    ],
)
def test_should_refuse_conflicting_h32_replay_cli_before_dispatch(monkeypatch, extra):
    for name in (
        "describe_minigrid_learnability_protocol",
        "run_minigrid_learnability_pilot",
        "run_minigrid_pilot",
    ):
        monkeypatch.setattr(cli, name, forbidden)
    monkeypatch.setattr(
        sys, "argv", ["pilot", *arguments(), "--old-a-rehearsal", *extra, "--describe"]
    )
    with pytest.raises(SystemExit) as failure:
        cli.main()
    assert failure.value.code == 2


def test_should_preserve_paid_incomplete_replay_report_and_nonzero_cli_exit(monkeypatch, capsys):
    report = dict(
        status="incomplete",
        reason="old-A replay native failure",
        work=dict(old_a_rehearsal=dict(native_train_calls=2, replay_rows=18, training_rows=9)),
    )
    monkeypatch.setattr(cli, "run_minigrid_learnability_pilot", lambda **options: report)
    monkeypatch.setattr(sys, "argv", ["pilot", *arguments(), "--old-a-rehearsal"])
    with pytest.raises(SystemExit) as failure:
        cli.main()
    assert failure.value.code == 2
    assert json.loads(capsys.readouterr().out) == report
