"""Explicit H32 S64 route guards and accounting without native learner work.

Forbidden constructors establish early admission. Factory/snapshot ports expose
the complete recipe and explicit width forwarding; numerical parity and the
original-data public run have their separately budgeted native checks.
"""

from dataclasses import asdict
import json
import sys

import numpy as np
import pytest

import scripts.run_coupled_attribution as cli
from src.adapters import minigrid_learnability as pilot
from src.adapters import minigrid_relaxation_control as control
from src.adapters.invariant_torch_feature_learner import InvariantTorchFeaturePolicy
from src.adapters.torch_feature_learner import TorchFeaturePolicy
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from test_minigrid_distinct_probe import raw_probe_data
from test_minigrid_invariant_learnability import episode
from test_minigrid_joint_catalogue_route import CLI_FLAGS, FLAGS, forbidden


METHODS = ("pc_relax64", "neutral_cpc_relax64")


def options(method="pc_relax64", **changes):
    return {
        **FLAGS,
        "qualification_method": method,
        "joint_catalogue": True,
        "wide_neural": True,
        **changes,
    }


@pytest.fixture
def no_resources(monkeypatch):
    for name in ("require_torch", "ProcessRssSampler", "MiniGridKeyDoor", "_history_learner"):
        monkeypatch.setattr(pilot, name, forbidden)
    monkeypatch.setattr(pilot.shared, "_train_arm", forbidden)
    monkeypatch.setattr(pilot.shared, "_collect_sources", forbidden)
    monkeypatch.setattr(pilot, "_collect_probe_sources", forbidden)


@pytest.mark.parametrize("method", METHODS)
def test_should_describe_actual_h32_s64_policy_and_original_isolated_doses(no_resources, method):
    result = pilot.describe_protocol(**options(method))
    native = method.removesuffix("_relax64")
    assert result["methods"] == [f"public_invariant_{method}"]
    assert result["hidden_dim"] == 32 and result["seed"] == 17
    assert result["inference_steps"] == 64 and result["inference_learning_rate"] == 0.04
    assert result["learning_rate"] == 0.03 and result["momentum"] == 0.9
    assert result["checkpoint_updates"] == [0, 32, 64, 96, 128]
    assert result["probe"]["source_stages"] == ["initial", "post_joint128"]
    assert result["primary_comparisons"] == []
    assert result["trainable_parameters"] == {f"public_invariant_{method}": 9735}
    assert result["public_history"]["parameters_by_method"] == result["trainable_parameters"]
    recipe = result["relaxation_control"]
    assert recipe["method"] == method and recipe["native_method"] == native
    assert recipe["original_inference_steps"] == 2
    assert recipe["native_policy"] == asdict(
        TorchFeaturePolicy(native, 296, 32, 17, inference_steps=64, num_classes=7)
    )
    wide = result["wide_neural"]
    assert wide["method"] == method and wide["native_method"] == native
    assert wide["inference_steps"] == 64
    assert wide["parameter_elements"] == 9735
    assert wide["parameter_array_equivalent_bytes"] == 38940
    assert wide["bp_optimizer_steps_derived"] == 0
    assert wide["pc_parameter_updates_derived"] == 192
    assert wide["nominal_latent_row_steps"] == 479232
    work = result["planned_work"]
    assert work["wide_neural"] == wide
    assert work["benchmark_native_train_calls"] == 128
    assert work["benchmark_training_rows"] == 6912
    assert work["probe_native_train_calls"] == work["return_native_train_calls"] == 32
    assert work["probe_training_rows"] == work["return_training_rows"] == 288
    assert work["native_train_calls"] == 192 and work["training_rows"] == 7488
    assert work["score_forwards"] == 94 and work["score_rows"] == 846
    assert work["final_episodes"] == work["replay_rows"] == 0
    assert work["relaxation_control"]["nominal_latent_iterations"] == 12288
    assert work["relaxation_control"]["nominal_latent_row_iterations"] == 479232


@pytest.mark.parametrize("method", METHODS)
def test_should_preserve_original_h8_control_when_width_is_omitted_or_false(no_resources, method):
    omitted = options(method)
    omitted.pop("wide_neural")
    result = pilot.describe_protocol(**omitted)
    assert result == pilot.describe_protocol(**omitted, wide_neural=False)
    assert result["hidden_dim"] == 8 and result["inference_steps"] == 64
    assert result["relaxation_control"]["native_policy"] == asdict(
        control.policy(f"public_invariant_{method}").native_policy
    )
    assert "wide_neural" not in result


@pytest.mark.parametrize("method", ["pc", "neutral_cpc"])
@pytest.mark.parametrize("wide", [False, True])
def test_should_preserve_original_s2_recipes_and_omit_control_metadata(no_resources, method, wide):
    result = pilot.describe_protocol(**options(method, wide_neural=wide))
    assert result["inference_steps"] == 2
    assert result["hidden_dim"] == (32 if wide else 8)
    assert "relaxation_control" not in result
    assert "relaxation_control" not in result["planned_work"]


@pytest.mark.parametrize("flag", [None, 0, 1, "true", [], np.bool_(True)])
def test_should_refuse_equal_valued_nonprimitive_width_flags_before_resources(no_resources, flag):
    for operation in (pilot.describe_protocol, pilot.run_pilot):
        with pytest.raises(ValueError):
            operation(**options(wide_neural=flag))


@pytest.mark.parametrize(
    "changed",
    [
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
                "weighted_bp",
                "weighted_pc",
                "weighted_catalogue",
                "train_action_margins",
                "pc_update_diagnostics",
                "activation_health",
                "gradient_geometry",
                "prior_gradient_geometry",
                "train_fit_bounds",
                "bp_update_diagnostics",
                "cue_relative_history",
                "memory_control",
                "acquisition_only",
                "train_a_rollout",
            )
        ),
        {"probe_inference": "fixed-weighted"},
        {"probe_inference": "bounded-weighted"},
        {"qualification_method": "public_invariant_pc_relax64"},
        {"qualification_method": "gated_cpc_relax64"},
        {"qualification_method": np.str_("pc_relax64")},
        {"joint_catalogue": 1},
        {"pc_update_diagnostics": 0},
    ],
)
def test_should_refuse_incomplete_or_mixed_public_control_before_resources(no_resources, changed):
    for operation in (pilot.describe_protocol, pilot.run_pilot):
        with pytest.raises(ValueError):
            operation(**options(**changed))


def arguments(method):
    result = [*CLI_FLAGS, "--joint-catalogue", "--wide-neural"]
    result[result.index("--qualification-method") + 1] = method
    return result


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("describe", [False, True])
def test_should_dispatch_explicit_control_and_width_to_existing_cli_route(
    monkeypatch, capsys, method, describe
):
    calls = []

    def selected(**values):
        calls.append(values)
        return {"status": "complete"}

    monkeypatch.setattr(
        cli, "describe_minigrid_learnability_protocol", selected if describe else forbidden
    )
    monkeypatch.setattr(cli, "run_minigrid_learnability_pilot", forbidden if describe else selected)
    monkeypatch.setattr(
        sys,
        "argv",
        ["pilot", *arguments(method)] + (["--describe"] if describe else []),
    )
    cli.main()
    assert calls == [options(method)]
    assert json.loads(capsys.readouterr().out) == {"status": "complete"}


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "extra, omitted",
    [
        ([], "--joint-catalogue"),
        ([], "--invariant-history"),
        ([], "--distinct-probe"),
        ([], "--old-task-readouts"),
        ([], "--return-to-a"),
        (["--old-a-rehearsal"], None),
        (["--extra-f-exposure"], None),
        (["--probe-rollouts"], None),
        (["--deferred-probe-rollouts"], None),
        (["--extended-wide-probe-budget"], None),
        (["--pc-update-diagnostics"], None),
        (["--native-activation-health"], None),
        (["--train-action-margins"], None),
        (["--weighted-pc"], None),
        (["--probe-inference", "fixed-weighted"], None),
    ],
)
def test_should_refuse_invalid_cli_control_before_dispatch(monkeypatch, method, extra, omitted):
    monkeypatch.setattr(cli, "describe_minigrid_learnability_protocol", forbidden)
    monkeypatch.setattr(cli, "run_minigrid_learnability_pilot", forbidden)
    selected = [value for value in arguments(method) if value != omitted]
    monkeypatch.setattr(sys, "argv", ["pilot", *selected, *extra, "--describe"])
    with pytest.raises(SystemExit) as failure:
        cli.main()
    assert failure.value.code == 2


@pytest.mark.parametrize("method", METHODS)
def test_should_construct_owned_h32_s64_policy_without_relabeling_the_outer_recipe(
    monkeypatch, method
):
    calls = []
    sentinel = object()

    def learner(policy):
        calls.append(policy)
        return sentinel

    monkeypatch.setattr(pilot, "InvariantTorchFeatureLearner", learner)
    monkeypatch.setattr(pilot, "TorchFeatureLearner", forbidden)
    assert pilot._history_learner(f"public_invariant_{method}", wide_neural=True) is sentinel
    assert calls == [
        InvariantTorchFeaturePolicy(
            TorchFeaturePolicy(
                method.removesuffix("_relax64"), 296, 32, 17, inference_steps=64, num_classes=7
            )
        )
    ]


@pytest.mark.parametrize("method", METHODS)
def test_should_forward_original_row_clock_and_explicit_width_to_snapshot_port(monkeypatch, method):
    calls = []
    source, owned = object(), object()

    def snapshot(state, point, bound, **values):
        calls.append((state, point, bound, values))
        return owned

    monkeypatch.setattr(pilot.shared, "_invariant_feature_snapshot", snapshot)
    bound = f"public_invariant_{method}"
    assert (
        pilot._invariant_snapshot(source, 128, bound, catalogue_rows=54, wide_neural=True) is owned
    )
    assert calls == [(source, 128, bound, {"catalogue_rows": 54, "wide_neural": True})]


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "changed",
    [
        {"catalogue_rows": None},
        {"catalogue_rows": 9},
        {"catalogue_rows": 54.0},
        {"wide_neural": 1},
        {"pc_update_diagnostics": True},
        {"activation_health": True},
        {"gradient_geometry": True},
        {"prior_gradient_geometry": True},
        {"train_action_margins": True},
        {"weighted_catalogue": True},
    ],
)
def test_should_refuse_private_readout_mixed_options_before_native_copy(
    monkeypatch, method, changed
):
    monkeypatch.setattr(pilot, "_invariant_snapshot", forbidden)
    values = {
        "method": f"public_invariant_{method}",
        "updates": 0,
        "sources": (),
        "budget": ToyBudgetSession(ToyExecutionBudget(max_training_updates=0), lambda: 1.0),
        "check_boundary": forbidden,
        "catalogue_rows": 54,
        "wide_neural": True,
        **changed,
    }
    with pytest.raises(ValueError):
        pilot._score_checkpoint(object(), object(), **values)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "changed",
    [
        {"old_a_rehearsal": True},
        {"extra_f_exposure": True},
        {"probe_rollouts": True},
        {"deferred": object()},
        {"extended_wide_probe_budget": True},
        {"probe_inference": "fixed-weighted"},
        {"paired": True},
        {"return_to_a": False},
        {"catalogue_rows": 9},
        {"catalogue_rows": 54.0},
        {"wide_neural": 1},
    ],
)
def test_should_refuse_private_probe_mixed_options_before_native_copy(monkeypatch, method, changed):
    data = tuple(pilot._history_packets(packets) for packets in raw_probe_data())
    monitors = pilot._history_packets(
        tuple(
            pilot._packet(episode("A", cue, "train", f"minigrid-source-A-{cue}"))
            for cue in ("blue", "red")
        )
    )
    monkeypatch.setattr(pilot, "_invariant_snapshot", forbidden)
    monkeypatch.setattr(pilot, "run_fresh_task_probe", forbidden)
    bound = f"public_invariant_{method}"
    values = {
        "sampler": None,
        "deadline": 1.0,
        "check_boundary": forbidden,
        "training_readouts": monitors,
        "return_to_a": True,
        "catalogue_rows": 54,
        "wide_neural": True,
        **changed,
    }
    with pytest.raises(ValueError):
        pilot._run_probe(
            {"checkpoint_updates": 0, "method": bound},
            {"row": {"method": bound}},
            object(),
            data,
            **values,
        )


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("unknown", [False, True])
def test_should_charge_actual_s64_work_without_certifying_unobserved_latent_steps(method, unknown):
    work = {
        "native_train_calls": 192,
        "training_rows": 7488,
        "failure_work_may_be_unobserved": unknown,
    }
    pilot._add_wide_neural_work(work, method)
    wide = work["wide_neural"]
    assert wide["method"] == method
    assert wide["native_method"] == method.removesuffix("_relax64")
    assert wide["inference_steps"] == 64
    assert wide["native_calls_reported"] == 192 and wide["training_rows_reported"] == 7488
    assert wide["bp_optimizer_steps_derived"] == wide["bp_backward_calls_derived"] == 0
    assert wide["pc_parameter_updates_derived"] == (None if unknown else 192)
    assert wide["nominal_latent_row_steps"] == (None if unknown else 479232)
    assert wide["failure_work_may_be_unobserved"] is unknown
