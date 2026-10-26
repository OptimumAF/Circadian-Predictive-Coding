"""Pure public F48 recipe, costs and CLI admission without learner resources."""

import json
import sys

import numpy as np
import pytest

import scripts.run_coupled_attribution as cli
from src.adapters import minigrid_f_exposure as recipe
from src.adapters import minigrid_learnability as pilot
from test_minigrid_joint_catalogue_route import CLI_FLAGS, FLAGS, METHODS


WIDE_FLAGS = {**FLAGS, "joint_catalogue": True, "wide_neural": True}
EXPOSURE_FLAGS = {**WIDE_FLAGS, "extra_f_exposure": True}


def forbidden(*args, **kwargs):
    raise AssertionError("F48 admission and description must precede all runtime resources")


@pytest.fixture(autouse=True)
def no_resources(monkeypatch):
    for name in ("require_torch", "ProcessRssSampler", "MiniGridKeyDoor", "_history_learner"):
        monkeypatch.setattr(pilot, name, forbidden)
    monkeypatch.setattr(pilot.shared, "_train_arm", forbidden)
    monkeypatch.setattr(pilot.shared, "_collect_sources", forbidden)
    monkeypatch.setattr(pilot, "_collect_probe_sources", forbidden)


@pytest.mark.parametrize("method", METHODS)
def test_should_describe_native_matched_f48_and_separate_original_a16_without_resources(method):
    options = {**WIDE_FLAGS, "qualification_method": method}
    original = pilot.describe_protocol(**options)
    selected = pilot.describe_protocol(**options, extra_f_exposure=True)
    assert original["protocol_id"] != selected["protocol_id"]
    assert "f48" in selected["protocol_id"].lower() or "extra_f_exposure" in selected["protocol_id"]
    assert selected["methods"] == original["methods"] == [f"public_invariant_{method}"]
    for field in ("seed", "hidden_dim", "learning_rate", "momentum", "checkpoint_updates"):
        assert selected[field] == original[field]
    assert selected["hidden_dim"] == 32 and selected["seed"] == 17
    assert selected["probe"]["source_updates"] == [0, 128]
    assert selected["probe"]["updates"] == 48
    assert selected["probe"]["checkpoint_updates"] == [0, 12, 24, 48]
    assert selected["return_phase"]["updates"] == 16
    assert selected["primary_comparisons"] == original["primary_comparisons"] == []
    work = selected["planned_work"]
    assert work["benchmark_native_train_calls"] == 128 and work["benchmark_training_rows"] == 6912
    assert work["probe_native_train_calls"] == 96 and work["probe_training_rows"] == 864
    assert work["native_train_calls"] == 256 and work["training_rows"] == 8064
    assert work["replay_rows"] == 0
    assert work["score_forwards"] == 94 and work["score_rows"] == 846
    assert work["final_episodes"] == work["learner_live_episodes"] == 0
    native = selected["wide_neural"]
    assert native["probe_native_calls"] == 96 and native["probe_training_rows"] == 864
    assert native["bp_optimizer_steps_derived"] == (256 if method == "bp" else 0)
    assert native["pc_parameter_updates_derived"] == (0 if method == "bp" else 256)
    assert native["nominal_latent_row_steps"] == (0 if method == "bp" else 16128)
    exposure = selected["extra_f_exposure"]
    assert exposure["input_training_batches"] == 16 and exposure["training_repetitions"] == 3
    assert exposure["updates_per_probe"] == 48 and exposure["training_rows_per_probe"] == 432
    assert exposure["checkpoint_updates"] == [0, 12, 24, 48]
    assert exposure["replay_logical_checkpoint_updates"] == [0, 4, 8, 16]
    assert exposure["replay_rows"] == exposure["bank_encoded_bytes"] == 0
    assert "old_a_rehearsal" not in selected


@pytest.mark.parametrize("options", [{}, FLAGS, WIDE_FLAGS])
def test_should_preserve_disabled_public_recipe_exactly(options):
    assert pilot.describe_protocol(**options) == pilot.describe_protocol(
        **options, extra_f_exposure=False
    )


@pytest.mark.parametrize("flag", [None, 0, 1, "true", np.bool_(True)])
def test_should_refuse_non_native_exposure_bool_before_resources(flag):
    for operation in (pilot.describe_protocol, pilot.run_pilot):
        with pytest.raises(ValueError, match="extra_f_exposure"):
            operation(**{**EXPOSURE_FLAGS, "extra_f_exposure": flag})
    with pytest.raises(ValueError, match="extra_f_exposure"):
        recipe.validate_option(flag, **WIDE_FLAGS)


@pytest.mark.parametrize(
    "changed",
    [
        {"public_history": False},
        {"invariant_history": False},
        {"joint_catalogue": False},
        {"wide_neural": False},
        {"distinct_probe": False},
        {"old_task_readouts": False},
        {"return_to_a": False},
        {"qualification_method": None},
        {"qualification_method": "pc_relax64"},
        {"qualification_method": "public_invariant_bp"},
        {"qualification_method": np.str_("bp")},
        {"pc_qualification": True},
        {"acquisition_only": True},
        {"memory_control": True},
        {"weighted_bp": True},
        {"weighted_pc": True},
        {"weighted_catalogue": True},
        {"probe_rollouts": True},
        {"deferred_probe_rollouts": True},
        {"extended_wide_probe_budget": True},
        {"old_a_rehearsal": True},
        {"train_a_rollout": True},
        {"activation_health": True},
        {"gradient_geometry": True},
        {"prior_gradient_geometry": True},
        {"pc_update_diagnostics": True},
        {"bp_update_diagnostics": True},
        {"probe_inference": "fixed-weighted"},
    ],
)
def test_should_refuse_incomplete_or_conflicting_public_f48_before_resources(changed):
    for operation in (pilot.describe_protocol, pilot.run_pilot):
        with pytest.raises(ValueError):
            operation(**{**EXPOSURE_FLAGS, **changed})


def arguments(method="bp"):
    result = [*CLI_FLAGS, "--joint-catalogue", "--wide-neural"]
    result[result.index("--qualification-method") + 1] = method
    return result


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("describe", [False, True])
def test_should_forward_only_the_explicit_full_public_f48_cli_recipe(
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
        ["pilot", *arguments(method), "--extra-f-exposure"] + (["--describe"] if describe else []),
    )
    cli.main()
    assert calls == [{**EXPOSURE_FLAGS, "qualification_method": method}]
    assert json.loads(capsys.readouterr().out) == {"status": "complete"}


@pytest.mark.parametrize("describe", [False, True])
def test_should_keep_original_cli_dispatch_kwargs_unchanged(monkeypatch, capsys, describe):
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
        ["--old-a-rehearsal"],
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
def test_should_refuse_conflicting_f48_cli_before_dispatch(monkeypatch, extra):
    for name in (
        "describe_minigrid_learnability_protocol",
        "run_minigrid_learnability_pilot",
        "run_minigrid_pilot",
    ):
        monkeypatch.setattr(cli, name, forbidden)
    monkeypatch.setattr(
        sys, "argv", ["pilot", *arguments(), "--extra-f-exposure", *extra, "--describe"]
    )
    with pytest.raises(SystemExit) as failure:
        cli.main()
    assert failure.value.code == 2


def test_should_preserve_honest_partial_f48_costs_and_nonzero_cli_exit(monkeypatch, capsys):
    report = {
        "status": "incomplete",
        "reason": "paid F48 update failed",
        "work": {"native_train_calls": 145, "training_rows": 7065},
    }
    monkeypatch.setattr(cli, "run_minigrid_learnability_pilot", lambda **options: report)
    monkeypatch.setattr(sys, "argv", ["pilot", *arguments(), "--extra-f-exposure"])
    with pytest.raises(SystemExit) as failure:
        cli.main()
    assert failure.value.code == 2
    assert json.loads(capsys.readouterr().out) == report
