"""H32 TRAIN diagnostic admission and costs; no native models or environments."""

import json
import sys

import numpy as np
import pytest

import scripts.run_coupled_attribution as cli
from src.adapters import minigrid_learnability as pilot
from test_minigrid_joint_catalogue_route import CLI_FLAGS, FLAGS, forbidden


OPTIONS = {
    **FLAGS,
    "qualification_method": "pc",
    "joint_catalogue": True,
    "wide_neural": True,
}


@pytest.fixture
def no_resources(monkeypatch):
    for name in ("require_torch", "ProcessRssSampler", "MiniGridKeyDoor", "_history_learner"):
        monkeypatch.setattr(pilot, name, forbidden)
    monkeypatch.setattr(pilot.shared, "_train_arm", forbidden)
    monkeypatch.setattr(pilot.shared, "_collect_sources", forbidden)
    monkeypatch.setattr(pilot, "_collect_probe_sources", forbidden)


@pytest.mark.parametrize("method", ["pc", "neutral_cpc"])
def test_should_declare_original_h32_recipe_and_separately_paid_train_diagnostics(
    no_resources, method
):
    options = {**OPTIONS, "qualification_method": method}
    original = pilot.describe_protocol(**options)
    assert original == pilot.describe_protocol(**options, pc_update_diagnostics=False)
    result = pilot.describe_protocol(**options, pc_update_diagnostics=True)
    for key in ("methods", "probe", "joint_catalogue", "wide_neural", "trainable_parameters"):
        assert result[key] == original[key]
    assert result["checkpoint_updates"] == [0, 32, 64, 96, 128]
    diagnostic = result["pc_update_diagnostics"]
    assert diagnostic["source_updates"] == [0, 128]
    assert diagnostic["native_recipe"] == {
        "method": method,
        "feature_dim": 296,
        "hidden_dim": 32,
        "num_classes": 7,
        "seed": 17,
        "learning_rate": 0.03,
        "momentum": 0.9,
        "inference_steps": 2,
        "inference_learning_rate": 0.04,
        "parameter_elements": 9735,
        "parameter_array_equivalent_bytes": 38940,
    }
    for key, value in original["planned_work"].items():
        assert result["planned_work"][key] == value
    work = result["planned_work"]["pc_update_diagnostics"]
    assert work["catalogue_calls"] == 2
    assert work["projection_rows"] == 108 and work["projection_array_bytes"] == 255744
    assert work["prior_forward_calls"] == 4 and work["prior_forward_rows"] == 216
    assert work["prior_backward_calls"] == 2 and work["prior_backward_rows"] == 108
    assert work["relaxation_forward_calls"] == 6 and work["relaxation_forward_rows"] == 324
    assert work["candidate_parameter_sets"] == 2
    assert work["native_train_calls"] == work["optimizer_steps"] == work["published_updates"] == 0
    assert result["planned_work"]["native_train_calls"] == 192
    assert result["planned_work"]["training_rows"] == 7488
    assert result["planned_work"]["score_forwards"] == 94
    assert result["planned_work"]["final_episodes"] == 0


@pytest.mark.parametrize("flag", [None, 0, 1, "true", [], np.bool_(True)])
def test_should_refuse_nonprimitive_diagnostic_flags_before_resources(no_resources, flag):
    for operation in (pilot.describe_protocol, pilot.run_pilot):
        with pytest.raises(ValueError):
            operation(**OPTIONS, pc_update_diagnostics=flag)


@pytest.mark.parametrize(
    "changed",
    [
        {"qualification_method": None},
        {"qualification_method": "bp"},
        {"qualification_method": "gated_cpc"},
        {"qualification_method": "pc_relax64"},
        {"qualification_method": np.str_("pc")},
        *(
            {key: False}
            for key in (
                "public_history",
                "invariant_history",
                "joint_catalogue",
                "distinct_probe",
                "old_task_readouts",
                "return_to_a",
            )
        ),
        *(
            {key: True}
            for key in (
                "old_a_rehearsal",
                "extra_f_exposure",
                "probe_rollouts",
                "deferred_probe_rollouts",
                "extended_wide_probe_budget",
                "weighted_pc",
                "weighted_catalogue",
                "train_action_margins",
                "activation_health",
                "gradient_geometry",
                "prior_gradient_geometry",
                "train_fit_bounds",
                "bp_update_diagnostics",
                "cue_relative_history",
                "memory_control",
            )
        ),
        {"probe_inference": "fixed-weighted"},
    ],
)
def test_should_refuse_mixed_or_incomplete_h32_diagnostics_before_resources(no_resources, changed):
    for operation in (pilot.describe_protocol, pilot.run_pilot):
        with pytest.raises(ValueError):
            operation(**{**OPTIONS, **changed}, pc_update_diagnostics=True)


def arguments(method="pc"):
    result = [*CLI_FLAGS, "--joint-catalogue", "--wide-neural"]
    result[result.index("--qualification-method") + 1] = method
    return result


@pytest.mark.parametrize("method", ["pc", "neutral_cpc"])
@pytest.mark.parametrize("describe", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_should_dispatch_only_the_explicit_cli_diagnostic_flag(
    monkeypatch, capsys, method, describe, enabled
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
        ["pilot", *arguments(method)]
        + (["--pc-update-diagnostics"] if enabled else [])
        + (["--describe"] if describe else []),
    )
    cli.main()
    assert calls == [
        {
            **OPTIONS,
            "qualification_method": method,
            **({"pc_update_diagnostics": True} if enabled else {}),
        }
    ]
    assert json.loads(capsys.readouterr().out) == {"status": "complete"}


@pytest.mark.parametrize("method", ["bp", "gated_cpc", "pc", "neutral_cpc"])
@pytest.mark.parametrize("describe", [False, True])
@pytest.mark.parametrize(
    "extra",
    [
        ["--old-a-rehearsal"],
        ["--extra-f-exposure"],
        ["--probe-rollouts"],
        ["--deferred-probe-rollouts"],
        ["--extended-wide-probe-budget"],
        ["--weighted-pc"],
        ["--native-activation-health"],
        ["--train-action-margins"],
    ],
)
def test_should_refuse_conflicting_cli_before_dispatch(monkeypatch, method, describe, extra):
    for name in ("describe_minigrid_learnability_protocol", "run_minigrid_learnability_pilot"):
        monkeypatch.setattr(cli, name, forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        ["pilot", *arguments(method), "--pc-update-diagnostics", *extra]
        + (["--describe"] if describe else []),
    )
    with pytest.raises(SystemExit) as failure:
        cli.main()
    assert failure.value.code == 2


@pytest.mark.parametrize("method", ["bp", "gated_cpc"])
def test_should_refuse_other_h32_methods_without_a_conflicting_cli_flag(monkeypatch, method):
    monkeypatch.setattr(cli, "describe_minigrid_learnability_protocol", forbidden)
    monkeypatch.setattr(
        sys, "argv", ["pilot", *arguments(method), "--pc-update-diagnostics", "--describe"]
    )
    with pytest.raises(SystemExit) as failure:
        cli.main()
    assert failure.value.code == 2


def test_should_preserve_h8_diagnostic_descriptor_without_h32_metadata(no_resources):
    options = {**OPTIONS, "wide_neural": False, "pc_update_diagnostics": True}
    result = pilot.describe_protocol(**options)
    assert result["hidden_dim"] == 8
    assert "native_recipe" not in result["pc_update_diagnostics"]
