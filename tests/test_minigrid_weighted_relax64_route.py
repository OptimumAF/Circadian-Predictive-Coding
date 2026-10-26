"""Weighted fixed64 public admission and isolation with scripted resource ports.

These checks exercise the existing public route, fixed recipe, transport and
probe proof without native learner construction. Full native state equality and
completed stream behavior have a separate serial integration gate.
"""

from dataclasses import asdict, replace
import json
import pickle
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import scripts.run_coupled_attribution as cli
from src.adapters import minigrid_joint_catalogue_route as joint
from src.adapters import minigrid_learnability as pilot
from src.adapters import minigrid_relaxation_control as control
from src.adapters import minigrid_weighted_catalogue_route as route
from src.adapters.invariant_torch_feature_learner import InvariantTorchFeatureSnapshot
from src.adapters.torch_catalogue_weighted import (
    TORCH_CATALOGUE_WEIGHTED_PC_ID,
    catalogue_class_weights,
)
from src.adapters.torch_feature_learner import TORCH_PC_ENERGY_ID
from test_minigrid_weighted_catalogue_route import (
    CLI_FLAGS,
    CLI_REQUIRED,
    FLAGS,
    forbidden,
    no_resources as no_resources,
    packets,
    receipt_arm,
)


METHODS = ("pc_relax64", "neutral_cpc_relax64")


def options(method="pc_relax64", **changes):
    return {**FLAGS, "qualification_method": method, "weighted_catalogue": True, **changes}


@pytest.mark.parametrize("method", METHODS)
def test_should_describe_original_fixed64_recipe_and_explicit_weighted_cost(no_resources, method):
    ordinary = pilot.describe_protocol(**options(method, weighted_catalogue=False))
    protocol = pilot.describe_protocol(**options(method))
    policy = control.policy(f"public_invariant_{method}")
    assert protocol["methods"] == [f"public_invariant_{method}"]
    assert protocol["inference_steps"] == 64
    assert protocol["relaxation_control"] == ordinary["relaxation_control"]
    assert protocol["relaxation_control"]["native_policy"] == asdict(policy.native_policy)
    for key in (
        "seed",
        "hidden_dim",
        "learning_rate",
        "momentum",
        "inference_learning_rate",
        "checkpoint_updates",
        "metric_direction",
        "joint_catalogue",
        "probe_support",
    ):
        assert protocol[key] == ordinary[key], key
    assert protocol["weighted_catalogue"]["method"] == method
    assert protocol["weighted_catalogue"]["benchmark_objective"] == TORCH_CATALOGUE_WEIGHTED_PC_ID
    assert protocol["probe"]["objective"] == TORCH_PC_ENERGY_ID
    assert protocol["return_phase"]["objective"] == TORCH_PC_ENERGY_ID
    assert "S64" in protocol["weighted_catalogue"]["exposure"]
    work = protocol["planned_work"]
    assert work["weighted_benchmark_native_train_calls"] == 128
    assert work["weighted_benchmark_training_rows"] == 6912
    assert work["weighted_benchmark_nominal_latent_row_iterations"] == 442368
    assert work["relaxation_control"]["nominal_latent_row_iterations"] == 479232
    assert (
        work["native_train_calls"],
        work["training_rows"],
        work["unweighted_probe_templates"],
    ) == (
        192,
        7488,
        2,
    )
    assert work["final_episodes"] == work["replay_rows"] == 0
    assert "pc_update_diagnostics" not in protocol
    assert "train_action_margins" not in protocol
    json.dumps(protocol, allow_nan=False)


@pytest.mark.parametrize("method", ("bp", "pc", "neutral_cpc"))
def test_should_keep_s2_weighted_metadata_and_false_default_distinct(no_resources, method):
    protocol = pilot.describe_protocol(**options(method))
    assert protocol["inference_steps"] == 2
    assert "relaxation_control" not in protocol
    assert "relaxation_control" not in protocol["planned_work"]
    assert protocol["planned_work"]["weighted_benchmark_nominal_latent_row_iterations"] == (
        0 if method == "bp" else 13824
    )
    without = {**FLAGS, "qualification_method": method}
    assert pilot.describe_protocol(**without) == pilot.describe_protocol(
        **without, weighted_catalogue=False
    )


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "changes",
    [
        dict(joint_catalogue=False),
        dict(distinct_probe=False),
        dict(old_task_readouts=False),
        dict(return_to_a=False),
        dict(weighted_bp=True),
        dict(weighted_pc=True),
        dict(train_action_margins=True),
        dict(pc_update_diagnostics=True),
        dict(train_action_margins=0),
        dict(pc_update_diagnostics=0),
        dict(weighted_catalogue=1),
        dict(weighted_catalogue=np.bool_(True)),
    ],
)
def test_should_refuse_unsupported_or_inexact_options_before_resources(
    no_resources, method, changes
):
    for operation in (pilot.describe_protocol, pilot.run_pilot):
        with pytest.raises(ValueError):
            operation(**options(method, **changes))


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("describe", (False, True))
def test_should_dispatch_explicit_weighted_fixed64_recipe(monkeypatch, capsys, method, describe):
    calls = []

    def selected(**settings):
        calls.append(settings)
        return {"status": "complete"}

    monkeypatch.setattr(
        cli, "describe_minigrid_learnability_protocol", selected if describe else forbidden
    )
    monkeypatch.setattr(cli, "run_minigrid_learnability_pilot", forbidden if describe else selected)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pilot",
            *CLI_FLAGS,
            method,
            *CLI_REQUIRED,
            "--weighted-catalogue",
            *(["--describe"] if describe else []),
        ],
    )
    cli.main()
    assert calls == [options(method)]
    assert json.loads(capsys.readouterr().out) == {"status": "complete"}


@pytest.mark.parametrize(
    "extra, missing",
    [
        (["--pc-update-diagnostics"], None),
        (["--train-action-margins"], None),
        (["--weighted-pc"], None),
        (["--weighted-bp"], None),
        ([], "--joint-catalogue"),
        ([], "--distinct-probe"),
    ],
)
def test_should_refuse_invalid_cli_before_dispatch(monkeypatch, extra, missing):
    monkeypatch.setattr(cli, "describe_minigrid_learnability_protocol", forbidden)
    monkeypatch.setattr(cli, "run_minigrid_learnability_pilot", forbidden)
    arguments = [*CLI_FLAGS, METHODS[0], *CLI_REQUIRED, "--weighted-catalogue", "--describe"]
    monkeypatch.setattr(sys, "argv", ["pilot", *(v for v in arguments if v != missing), *extra])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


@pytest.mark.parametrize("method", METHODS)
def test_should_bind_original_catalogue_and_all_transports_to_named_policy(monkeypatch, method):
    sources = packets()
    plan = joint.compose(sources)
    saved = pickle.dumps((sources, plan.training))
    calls = []
    sentinel = object()

    def constructor(policy, catalogue, updates):
        calls.append((policy, catalogue, updates))
        return sentinel

    monkeypatch.setattr(route.TorchTrainWeightedLearner, "from_catalogue", constructor)
    build = route.factory(sources, plan.training, method=method)
    assert calls == []
    assert build(f"public_invariant_{method}") is sentinel
    policy, catalogue, updates = calls[0]
    assert policy == control.policy(f"public_invariant_{method}")
    assert policy.native_policy.method == method.removesuffix("_relax64")
    assert policy.native_policy.inference_steps == 64 and updates == 128
    assert catalogue.episodes == tuple(item.episode for item in sources)
    assert all(a is not b.episode for a, b in zip(catalogue.episodes, sources, strict=True))
    assert catalogue.source_spans == plan.source_spans
    assert catalogue_class_weights(catalogue) == (1.8, 1.8, 0.36, 1.8, 1.0, 1.8, 1.0)
    assert pickle.dumps((sources, plan.training)) == saved
    with pytest.raises(ValueError):
        build(f"public_invariant_{method.removesuffix('_relax64')}")
    assert len(calls) == 1


@pytest.mark.parametrize("fault", ("late_rows", "late_targets", "source_order", "source_identity"))
def test_should_refuse_changed_schedule_or_source_before_fixed64_constructor(no_resources, fault):
    sources = packets()
    training = joint.compose(sources).training
    if fault == "source_order":
        sources = (sources[1], sources[0], *sources[2:])
    elif fault == "source_identity":
        sources = (
            replace(sources[0], episode=replace(sources[0].episode, episode_id="other")),
            *sources[1:],
        )
    else:
        batch = training[-1]
        if fault == "late_rows":
            batch = replace(batch, row_ids=training[0].row_ids)
        else:
            targets = batch.targets.copy()
            targets[-1, 0] = (targets[-1, 0] + 1) % 7
            batch = replace(batch, targets=targets)
        training = (*training[:-1], batch)
    with pytest.raises(ValueError):
        route.factory(sources, training, method=METHODS[0])


@pytest.mark.parametrize("method", METHODS)
def test_should_require_public_recipe_receipts_with_existing_native_objective(method):
    arm = receipt_arm(method)
    route.check_receipts(arm, method=method)
    arm["row"]["method"] = f"public_invariant_{method.removesuffix('_relax64')}"
    with pytest.raises(ValueError):
        route.check_receipts(arm, method=method)
    arm["row"]["method"] = f"public_invariant_{method}"
    arm["row"]["stream_result"]["applied_updates"][-1]["diagnostic"]["definition"] = (
        TORCH_PC_ENERGY_ID
    )
    with pytest.raises(ValueError):
        route.check_receipts(arm, method=method)


def probe_fixture(monkeypatch, method=METHODS[0]):
    source: Any = object.__new__(route.TorchTrainWeightedLearner)
    source._catalogue = object()
    source._catalogue_updates = 128
    source._catalogue_rows = 54
    native: Any = SimpleNamespace(completed_updates=0, training_examples=0)
    before = InvariantTorchFeatureSnapshot(control.policy(f"public_invariant_{method}"), native)
    monkeypatch.setattr(route.TorchTrainWeightedLearner, "snapshot_state", lambda self: before)
    return source, before, dict(source=source, row=dict(method=f"public_invariant_{method}"))


@pytest.mark.parametrize("method", METHODS)
def test_should_prove_full_zero_state_before_ordinary_fixed64_probe(monkeypatch, method):
    source, before, arm = probe_fixture(monkeypatch, method)
    template = SimpleNamespace(snapshot_state=lambda: before)
    policies, comparisons = [], []

    def construct(policy):
        policies.append(policy)
        return template

    def same(left, right):
        comparisons.append((left, right))
        return left is before and right is before

    monkeypatch.setattr(route, "InvariantTorchFeatureLearner", construct)
    monkeypatch.setattr(route, "_same_state", same)
    monkeypatch.setattr(route.online, "probe_arm", forbidden)
    result = route.probe_arm(arm, method=method)
    assert result["source"] is template and arm["source"] is source
    assert result["row"] is arm["row"]
    assert policies == [before.policy] and comparisons == [(before, before), (before, before)]


@pytest.mark.parametrize(
    "fault",
    (
        "s2_policy",
        "other_base",
        "completed",
        "examples",
        "bool_counter",
        "owner",
        "method",
        "catalogue",
        "schedule",
        "rows",
        "rows_bool",
        "snapshot_type",
    ),
)
def test_should_refuse_changed_probe_recipe_or_owner_before_template(monkeypatch, fault):
    source, before, arm = probe_fixture(monkeypatch)
    monkeypatch.setattr(route, "InvariantTorchFeatureLearner", forbidden)
    if fault in ("s2_policy", "other_base"):
        native = before.policy.native_policy
        changed = replace(
            native,
            **({"inference_steps": 2} if fault == "s2_policy" else {"method": "neutral_cpc"}),
        )
        object.__setattr__(before, "policy", replace(before.policy, native_policy=changed))
    elif fault == "completed":
        before.native_state.completed_updates = 1
    elif fault == "examples":
        before.native_state.training_examples = 54
    elif fault == "bool_counter":
        before.native_state.completed_updates = False
    elif fault == "owner":
        arm["source"] = object()
    elif fault == "method":
        arm["row"]["method"] = "public_invariant_pc"
    elif fault == "catalogue":
        source._catalogue = None
    elif fault == "schedule":
        source._catalogue_updates = 127
    elif fault in ("rows", "rows_bool"):
        source._catalogue_rows = True if fault == "rows_bool" else 9
    else:
        monkeypatch.setattr(
            route.TorchTrainWeightedLearner,
            "snapshot_state",
            lambda self: SimpleNamespace(
                policy=before.policy, completed_updates=0, training_examples=0
            ),
        )
    with pytest.raises(ValueError):
        route.probe_arm(arm, method=METHODS[0])


@pytest.mark.parametrize("fault", ("template_state", "source_state", "source_owner"))
def test_should_refuse_unproved_template_or_source_change_after_construction(monkeypatch, fault):
    _source, before, arm = probe_fixture(monkeypatch)
    different = object()
    template = SimpleNamespace(
        snapshot_state=lambda: different if fault == "template_state" else before
    )

    def construct(policy):
        if fault == "source_owner":
            arm["source"] = object()
        elif fault == "source_state":
            monkeypatch.setattr(
                route.TorchTrainWeightedLearner, "snapshot_state", lambda self: different
            )
        return template

    monkeypatch.setattr(route, "InvariantTorchFeatureLearner", construct)
    monkeypatch.setattr(
        route, "_same_state", lambda left, right: left is before and right is before
    )
    with pytest.raises(ValueError):
        route.probe_arm(arm, method=METHODS[0])
