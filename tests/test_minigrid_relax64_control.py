"""Fixed relaxation recipes use existing routes without constructing learners.

Native complete-route and neutral-assignment parity checks live separately. Here
resource sentinels verify early admission, public naming, recipe and row binding.
"""

from dataclasses import asdict, replace
import json
import sys
from types import SimpleNamespace
from typing import Any, cast

import pytest

import scripts.run_coupled_attribution as cli
from src.adapters import minigrid_learnability as pilot
from src.adapters import minigrid_learning_pilot as shared
from src.adapters import minigrid_relaxation_control as control
from src.adapters.invariant_torch_feature_learner import InvariantTorchFeaturePolicy
from src.adapters.torch_feature_learner import TorchFeaturePolicy
from test_minigrid_joint_catalogue_route import CLI_FLAGS, FLAGS, forbidden
from test_minigrid_joint_catalogue_route import no_resources as no_resources


METHODS = ("pc_relax64", "neutral_cpc_relax64")


def options(method: str, **changes: Any) -> dict[str, Any]:
    return {**FLAGS, "qualification_method": method, "joint_catalogue": True, **changes}


@pytest.mark.parametrize("method", METHODS)
def test_should_bind_distinct_outer_name_to_unchanged_native_algorithm(method):
    bound = f"public_invariant_{method}"
    policy = control.policy(bound)
    native_method = method.removesuffix("_relax64")
    assert type(policy) is InvariantTorchFeaturePolicy
    assert policy.native_policy == TorchFeaturePolicy(
        native_method, 296, 8, 17, inference_steps=64, num_classes=7
    )
    assert policy.method == f"public_invariant_{native_method}"
    assert bound != policy.method
    assert control.METHODS == METHODS
    assert control.BOUND_METHODS == tuple(f"public_invariant_{value}" for value in METHODS)


@pytest.mark.parametrize("method", ("pc", "public_invariant_pc", "gated_cpc_relax64", None))
def test_should_refuse_non_control_policy_names(method):
    with pytest.raises(ValueError):
        control.policy(method)


@pytest.mark.parametrize("method", METHODS)
def test_should_describe_full_original_dose_with_explicit_extra_inference(no_resources, method):
    protocol = pilot.describe_protocol(**options(method))
    assert protocol["methods"] == [f"public_invariant_{method}"]
    assert protocol["inference_steps"] == 64
    assert protocol["seed"] == 17 and protocol["hidden_dim"] == 8
    assert protocol["learning_rate"] == 0.03 and protocol["momentum"] == 0.9
    assert protocol["inference_learning_rate"] == 0.04
    assert protocol["checkpoint_updates"] == [0, 32, 64, 96, 128]
    assert protocol["joint_catalogue"]["training_row_exposures"] == 6912
    recipe = protocol["relaxation_control"]
    assert recipe["native_method"] == method.removesuffix("_relax64")
    assert recipe["original_inference_steps"] == 2
    assert recipe["native_policy"] == asdict(
        control.policy(f"public_invariant_{method}").native_policy
    )
    work = protocol["planned_work"]
    assert work["native_train_calls"] == 192 and work["training_rows"] == 7488
    assert work["final_episodes"] == 0 and work["replay_rows"] == 0
    assert work["relaxation_control"]["nominal_latent_iterations"] == 192 * 64
    assert work["relaxation_control"]["nominal_latent_row_iterations"] == 7488 * 64


@pytest.mark.parametrize("method", ("pc", "neutral_cpc"))
def test_should_preserve_original_s2_recipe_and_wire_without_control_fields(no_resources, method):
    protocol = pilot.describe_protocol(**options(method))
    assert protocol["methods"] == [f"public_invariant_{method}"]
    assert protocol["inference_steps"] == 2
    assert "relaxation_control" not in protocol
    assert "relaxation_control" not in protocol["planned_work"]


@pytest.mark.parametrize(
    "changes",
    [
        dict(joint_catalogue=False),
        dict(distinct_probe=False),
        dict(old_task_readouts=False),
        dict(return_to_a=False),
        dict(train_action_margins=True),
        dict(pc_update_diagnostics=True),
        dict(weighted_bp=True),
        dict(acquisition_only=True),
    ],
)
def test_should_refuse_unsupported_public_combinations_before_resources(no_resources, changes):
    selected = options(METHODS[0], **changes)
    with pytest.raises(ValueError):
        pilot.describe_protocol(**selected)
    with pytest.raises(ValueError):
        pilot.run_pilot(**selected)


@pytest.mark.parametrize("describe", (False, True))
@pytest.mark.parametrize("method", METHODS)
def test_should_forward_distinct_cli_recipe_to_existing_route(
    monkeypatch, capsys, describe, method
):
    calls = []

    def selected(**kwargs):
        calls.append(kwargs)
        return dict(status="complete")

    monkeypatch.setattr(
        cli, "describe_minigrid_learnability_protocol", selected if describe else forbidden
    )
    monkeypatch.setattr(cli, "run_minigrid_learnability_pilot", forbidden if describe else selected)
    arguments = [method if value == "bp" else value for value in CLI_FLAGS]
    monkeypatch.setattr(
        sys,
        "argv",
        ["pilot", *arguments, "--joint-catalogue"] + (["--describe"] if describe else []),
    )
    cli.main()
    assert calls == [options(method)]
    assert json.loads(capsys.readouterr().out) == dict(status="complete")


@pytest.mark.parametrize(
    "extra, omitted",
    [
        ([], "--joint-catalogue"),
        ([], "--distinct-probe"),
        ([], "--old-task-readouts"),
        ([], "--return-to-a"),
        (["--train-action-margins"], None),
        (["--pc-update-diagnostics"], None),
    ],
)
def test_should_refuse_invalid_cli_control_before_dispatch(monkeypatch, extra, omitted):
    monkeypatch.setattr(cli, "describe_minigrid_learnability_protocol", forbidden)
    monkeypatch.setattr(cli, "run_minigrid_learnability_pilot", forbidden)
    arguments = [METHODS[0] if value == "bp" else value for value in CLI_FLAGS]
    arguments = [value for value in [*arguments, "--joint-catalogue"] if value != omitted]
    monkeypatch.setattr(sys, "argv", ["pilot", *arguments, *extra, "--describe"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def copied_state(monkeypatch, policy):
    state = SimpleNamespace(policy=policy, completed_updates=2, training_examples=108)
    monkeypatch.setattr(shared, "copy_invariant_torch_feature_snapshot", lambda value: state)
    return state


@pytest.mark.parametrize("method", METHODS)
def test_should_admit_only_complete_64_step_recipe_with_explicit_54_row_clock(monkeypatch, method):
    bound = f"public_invariant_{method}"
    state = copied_state(monkeypatch, control.policy(bound))
    shared._invariant_feature_counter(state, 2, bound, catalogue_rows=54)
    for rows in (None, 9, 54.0, True):
        with pytest.raises(ValueError):
            shared._invariant_feature_counter(state, 2, bound, catalogue_rows=cast(Any, rows))
    state.training_examples = 18
    with pytest.raises(ValueError):
        shared._invariant_feature_counter(state, 2, bound, catalogue_rows=54)


@pytest.mark.parametrize(
    "changes",
    [
        dict(inference_steps=2),
        dict(method="neutral_cpc"),
        dict(seed=18),
        dict(hidden_dim=9),
        dict(learning_rate=0.031),
        dict(momentum=0.8),
        dict(inference_learning_rate=0.041),
    ],
)
def test_should_refuse_cross_recipe_or_changed_native_policy(monkeypatch, changes):
    bound = "public_invariant_pc_relax64"
    policy = control.policy(bound)
    state = copied_state(
        monkeypatch, replace(policy, native_policy=replace(policy.native_policy, **changes))
    )
    with pytest.raises(ValueError):
        shared._invariant_feature_counter(state, 2, bound, catalogue_rows=54)


@pytest.mark.parametrize("method", METHODS)
def test_should_refuse_relabeling_64_step_state_as_original_s2_baseline(monkeypatch, method):
    state = copied_state(monkeypatch, control.policy(f"public_invariant_{method}"))
    old_method = f"public_invariant_{method.removesuffix('_relax64')}"
    with pytest.raises(ValueError):
        shared._invariant_feature_counter(state, 2, old_method, catalogue_rows=54)


@pytest.mark.parametrize("method", METHODS)
def test_should_construct_existing_bound_learner_with_exact_control_policy(monkeypatch, method):
    seen = []
    sentinel = object()

    def learner(policy):
        seen.append(policy)
        return sentinel

    monkeypatch.setattr(pilot, "InvariantTorchFeatureLearner", learner)
    assert pilot._history_learner(f"public_invariant_{method}") is sentinel
    assert seen == [control.policy(f"public_invariant_{method}")]


@pytest.mark.parametrize(
    "changes",
    [dict(joint_catalogue=1), dict(train_action_margins=0), dict(pc_update_diagnostics=0)],
)
def test_should_refuse_equal_valued_non_bool_control_flags_before_resources(no_resources, changes):
    with pytest.raises(ValueError):
        pilot.describe_protocol(**options(METHODS[0], **changes))


@pytest.mark.parametrize(
    "calls, rows, unknown", [(192, 7488, False), (128, 6912, True), (0, 0, True)]
)
def test_should_charge_returned_iterations_without_hiding_unobserved_failure(calls, rows, unknown):
    work = dict(
        native_train_calls=calls,
        training_rows=rows,
        failure_work_may_be_unobserved=unknown,
    )
    control.add_work(work, [dict(method="public_invariant_pc_relax64")])
    measured = work["relaxation_control"]
    assert measured["inference_steps"] == 64
    assert measured["native_train_calls_returned"] == calls
    assert measured["nominal_latent_iterations_returned"] == 64 * calls
    assert measured["nominal_latent_row_iterations_returned"] == 64 * rows
    assert measured["failure_work_may_be_unobserved"] is unknown
    assert "unreturned" in measured["scope"] and "unknown" in measured["scope"]


def test_should_leave_original_work_wire_unchanged():
    work = dict(native_train_calls=192, training_rows=7488, failure_work_may_be_unobserved=False)
    saved = work.copy()
    control.add_work(work, [dict(method="public_invariant_pc")])
    assert work == saved


def test_should_refuse_mixing_fixed64_costs_with_unmatched_methods():
    with pytest.raises(ValueError):
        control.add_work(
            dict(native_train_calls=384, training_rows=14976, failure_work_may_be_unobserved=False),
            [dict(method="public_invariant_pc_relax64"), dict(method="public_invariant_pc")],
        )
