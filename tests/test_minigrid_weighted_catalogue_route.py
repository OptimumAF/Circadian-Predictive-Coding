"""Public weighted catalogue admission with scripted ports and no native work.

Complete synthetic TRAIN sources exercise metadata/array admission. Description
and CLI tests forbid collection, native learners, training, and real games.
"""

from dataclasses import replace
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
from src.adapters import minigrid_weighted_catalogue_route as route
from src.adapters.torch_catalogue_weighted import (
    TORCH_CATALOGUE_WEIGHTED_CE_ID,
    TORCH_CATALOGUE_WEIGHTED_PC_ID,
)
from src.adapters.torch_feature_learner import TORCH_CE_ID, TORCH_PC_ENERGY_ID
from test_minigrid_joint_catalogue_route import packets


METHODS = ("bp", "pc", "neutral_cpc")
FLAGS: dict[str, Any] = dict(
    public_history=True,
    invariant_history=True,
    qualification_method="bp",
    distinct_probe=True,
    old_task_readouts=True,
    return_to_a=True,
    joint_catalogue=True,
)
CLI_FLAGS = [
    "--scenario",
    "torch-minigrid-history-learnability",
    "--invariant-history",
    "--qualification-method",
]
CLI_REQUIRED = [
    "--distinct-probe",
    "--old-task-readouts",
    "--return-to-a",
    "--joint-catalogue",
]


def forbidden(*args, **kwargs):
    raise AssertionError("pure admission must not construct or execute native resources")


@pytest.fixture
def no_resources(monkeypatch):
    for name in ("require_torch", "ProcessRssSampler", "MiniGridKeyDoor", "_history_learner"):
        monkeypatch.setattr(pilot, name, forbidden)
    monkeypatch.setattr(pilot.shared, "_train_arm", forbidden)
    monkeypatch.setattr(pilot.shared, "_collect_sources", forbidden)
    monkeypatch.setattr(route.TorchTrainWeightedLearner, "from_catalogue", forbidden)


def options(method="bp"):
    return {**FLAGS, "qualification_method": method}


@pytest.mark.parametrize("settings", [{}, dict(public_history=True), FLAGS])
def test_should_preserve_original_protocol_when_catalogue_weighting_is_false(
    no_resources, settings
):
    assert pilot.describe_protocol(**settings) == pilot.describe_protocol(
        **settings, weighted_catalogue=False
    )


@pytest.mark.parametrize("method", METHODS)
def test_should_describe_distinct_objective_with_original_joint_dose_and_metrics(
    no_resources, method
):
    ordinary = pilot.describe_protocol(**options(method))
    protocol = pilot.describe_protocol(**options(method), weighted_catalogue=True)
    objective = TORCH_CATALOGUE_WEIGHTED_CE_ID if method == "bp" else TORCH_CATALOGUE_WEIGHTED_PC_ID
    probe_objective = TORCH_CE_ID if method == "bp" else TORCH_PC_ENERGY_ID
    assert protocol["protocol_id"] != ordinary["protocol_id"]
    assert protocol["methods"] == [f"public_invariant_{method}"]
    assert protocol["weighted_catalogue"]["benchmark_objective"] == objective
    assert protocol["weighted_catalogue"]["probe_objective"] == probe_objective
    assert protocol["probe"]["objective"] == probe_objective
    assert protocol["return_phase"]["objective"] == probe_objective
    for key in (
        "seed",
        "hidden_dim",
        "learning_rate",
        "momentum",
        "inference_steps",
        "inference_learning_rate",
        "checkpoint_updates",
        "metric_direction",
        "probe_support",
        "joint_catalogue",
    ):
        assert protocol[key] == ordinary[key]
    work = protocol["planned_work"]
    assert work["benchmark_native_train_calls"] == 128
    assert work["benchmark_training_rows"] == 6912
    assert work["native_train_calls"] == 192 and work["training_rows"] == 7488
    assert work["weighted_benchmark_native_train_calls"] == 128
    assert work["weighted_benchmark_training_rows"] == 6912
    assert work["weighted_benchmark_nominal_latent_row_iterations"] == (
        0 if method == "bp" else 13824
    )
    assert work["unweighted_probe_templates"] == 2
    assert work["final_episodes"] == work["replay_rows"] == 0
    assert protocol["primary_comparisons"] == [] and "differences" not in protocol
    json.dumps(protocol, allow_nan=False)


@pytest.mark.parametrize("value", [None, 0, 1, "true", np.bool_(True)])
def test_should_refuse_inexact_weighting_flags_before_resources(no_resources, value):
    with pytest.raises(ValueError):
        pilot.describe_protocol(**FLAGS, weighted_catalogue=value)
    with pytest.raises(ValueError):
        pilot.run_pilot(**FLAGS, weighted_catalogue=value)


@pytest.mark.parametrize(
    "changed",
    [
        dict(public_history=False),
        dict(public_history=1),
        dict(invariant_history=False),
        dict(joint_catalogue=False),
        dict(distinct_probe=False),
        dict(old_task_readouts=False),
        dict(return_to_a=False),
        dict(qualification_method=None),
        dict(qualification_method="gated_cpc"),
        # Named PC/neutral fixed64 is now supported; active fixed64 is not.
        dict(qualification_method="gated_cpc_relax64"),
        dict(pc_qualification=True),
        dict(acquisition_only=True),
        dict(weighted_bp=True),
        dict(qualification_method="pc", weighted_pc=True),
        dict(memory_control=True),
        dict(activation_health=True),
        dict(gradient_geometry=True),
        dict(prior_gradient_geometry=True),
        dict(probe_rollouts=True),
        dict(deferred_probe_rollouts=True),
        dict(pc_update_diagnostics=True),
    ],
)
def test_should_refuse_incompatible_routes_before_resources(no_resources, changed):
    settings = {**FLAGS, **changed, "weighted_catalogue": True}
    with pytest.raises(ValueError):
        pilot.describe_protocol(**settings)
    with pytest.raises(ValueError):
        pilot.run_pilot(**settings)


@pytest.mark.parametrize("method", METHODS)
def test_should_bind_owned_complete_catalogue_and_fixed_policy_once(monkeypatch, method):
    sources = packets()
    plan = joint.compose(sources)
    saved = pickle.dumps((sources, plan.training))
    calls = []

    def constructor(policy, catalogue, updates):
        calls.append((policy, catalogue, updates))
        return "owned"

    monkeypatch.setattr(route.TorchTrainWeightedLearner, "from_catalogue", constructor)
    build = route.factory(sources, plan.training, method=method)
    assert calls == []
    assert build(f"public_invariant_{method}") == "owned"
    assert len(calls) == 1
    policy, catalogue, updates = calls[0]
    native = policy.native_policy
    assert (native.method, policy.feature_dim, policy.hidden_dim, policy.num_classes) == (
        method,
        296,
        8,
        7,
    )
    assert (native.seed, native.learning_rate, native.momentum) == (17, 0.03, 0.9)
    assert (native.inference_steps, native.inference_learning_rate) == (2, 0.04)
    assert updates == 128
    assert catalogue.episodes == tuple(item.episode for item in sources)
    assert all(a is not b.episode for a, b in zip(catalogue.episodes, sources, strict=True))
    assert catalogue.source_spans == plan.source_spans
    assert [(span.start, span.stop) for span in catalogue.source_spans] == [
        (i * 9, (i + 1) * 9) for i in range(6)
    ]
    assert pickle.dumps((sources, plan.training)) == saved


@pytest.mark.parametrize("method", ["pc", "public_invariant_bp", None, 1])
def test_should_refuse_wrong_bound_method_before_constructor(no_resources, method):
    sources = packets()
    with pytest.raises(ValueError):
        route.factory(sources, joint.compose(sources).training, method="neutral_cpc")(method)


@pytest.mark.parametrize(
    "fault",
    [
        "list",
        "short",
        "order",
        "missing_episode",
        "context",
        "role",
        "classes",
        "features",
        "targets",
        "dtype",
        "identity",
        "next_geometry",
    ],
)
def test_should_refuse_changed_original_sources_before_constructor(no_resources, fault):
    sources: Any = packets()
    training = joint.compose(sources).training
    if fault == "list":
        sources = list(sources)
    elif fault == "short":
        sources = sources[:-1]
    elif fault == "order":
        sources = (sources[1], sources[0], *sources[2:])
    else:
        first = sources[0]
        record = first.episode
        if fault == "missing_episode":
            first = replace(first, episode=None)
        elif fault == "context":
            first = replace(first, context="B")
        elif fault == "role":
            first = replace(first, role="probe_train")
        elif fault == "classes":
            first = replace(first, num_classes=8)
        elif fault == "features":
            values = first.features.copy()
            values[0, 0] += 1
            first = replace(first, features=values)
        elif fault == "targets":
            values = first.targets.copy()
            values[0, 0] = (values[0, 0] + 1) % 7
            first = replace(first, targets=values)
        elif fault == "dtype":
            first = replace(first, features=first.features.astype(np.float32))
        elif fault == "identity":
            first = replace(first, episode=replace(record, episode_id="replaced"))
        else:
            transitions = (
                *record.transitions[:-1],
                replace(record.transitions[-1], next_observation=(0.0,) * 296),
            )
            first = replace(first, episode=replace(record, transitions=transitions))
        sources = (first, *sources[1:])
    with pytest.raises(ValueError):
        route.factory(sources, training, method="pc")("public_invariant_pc")


@pytest.mark.parametrize(
    "fault",
    [
        "list",
        "short",
        "role",
        "context",
        "classes",
        "row_ids",
        "features",
        "targets",
        "dtype",
        "episode",
    ],
)
def test_should_bind_every_transport_wake_to_exact_original_catalogue(no_resources, fault):
    sources = packets()
    training: Any = joint.compose(sources).training
    if fault == "list":
        training = list(training)
    elif fault == "short":
        training = training[:-1]
    else:
        # Corrupt a late wake so admission must inspect the complete schedule.
        batch = training[-1]
        if fault == "role":
            batch = replace(batch, role="probe_train")
        elif fault == "context":
            batch = replace(batch, context="A")
        elif fault == "classes":
            batch = replace(batch, num_classes=8)
        elif fault == "row_ids":
            batch = replace(batch, row_ids=training[0].row_ids)
        elif fault in ("features", "targets"):
            values = getattr(batch, fault).copy()
            values[-1, 0] += 1
            batch = replace(batch, **{fault: values})
        elif fault == "dtype":
            batch = replace(batch, features=batch.features.astype(np.float32))
        else:
            batch = replace(batch, episode=sources[0].episode)
        training = (*training[:-1], batch)
    with pytest.raises(ValueError):
        route.factory(sources, training, method="pc")("public_invariant_pc")


def receipt_arm(method="bp") -> dict[str, Any]:
    objective = TORCH_CATALOGUE_WEIGHTED_CE_ID if method == "bp" else TORCH_CATALOGUE_WEIGHTED_PC_ID
    return dict(
        row=dict(
            method=f"public_invariant_{method}",
            stream_result=dict(
                evaluation_scope="training_only",
                completed_updates=128,
                training_examples=6912,
                applied_updates=[
                    dict(update_number=i, diagnostic=dict(definition=objective, value=0.25))
                    for i in range(1, 129)
                ],
            ),
        )
    )


@pytest.mark.parametrize("method", METHODS)
def test_should_accept_only_paid_catalogue_objective_receipts(method):
    route.check_receipts(receipt_arm(method), method=method)


@pytest.mark.parametrize(
    "fault",
    [
        "scope",
        "count",
        "count_bool",
        "rows",
        "short",
        "counter",
        "definition",
        "nonfinite",
        "missing",
        "method",
    ],
)
def test_should_refuse_corrupted_catalogue_receipts(fault):
    arm = receipt_arm("pc")
    result = arm["row"]["stream_result"]
    receipt = result["applied_updates"][0]
    if fault == "scope":
        result["evaluation_scope"] = "final"
    elif fault == "count":
        result["completed_updates"] = 127
    elif fault == "count_bool":
        result["completed_updates"] = True
    elif fault == "rows":
        result["training_examples"] = 1152
    elif fault == "short":
        result["applied_updates"].pop()
    elif fault == "counter":
        receipt["update_number"] = True
    elif fault == "definition":
        receipt["diagnostic"]["definition"] = TORCH_PC_ENERGY_ID
    elif fault == "nonfinite":
        receipt["diagnostic"]["value"] = float("nan")
    elif fault == "missing":
        result["applied_updates"][0] = None
    else:
        arm["row"]["method"] = "public_invariant_bp"
    with pytest.raises(ValueError):
        route.check_receipts(arm, method="pc")


def probe_source(monkeypatch, method="bp"):
    # Admission requires the exact owner, but its native constructor/snapshot is
    # replaced by a bounded scripted port. Native equality is a separate gate.
    source: Any = object.__new__(route.TorchTrainWeightedLearner)
    source._catalogue = object()
    source._catalogue_updates = 128
    source._catalogue_rows = 54
    before = SimpleNamespace(
        policy=route.online._fixed_policy(method),
        completed_updates=0,
        training_examples=0,
    )
    monkeypatch.setattr(route.TorchTrainWeightedLearner, "snapshot_state", lambda self: before)
    return source, before, dict(source=source, row=dict(method=f"public_invariant_{method}"))


@pytest.mark.parametrize("method", METHODS)
def test_should_delegate_only_fixed_zero_update_catalogue_to_ordinary_probe_proof(
    monkeypatch, method
):
    source, before, arm = probe_source(monkeypatch, method)
    calls = []
    ordinary = object()

    def checked(value, *, method):
        assert value is arm and value["source"] is source
        calls.append(method)
        return {**value, "source": ordinary}

    monkeypatch.setattr(route.online, "probe_arm", checked)
    result = route.probe_arm(arm, method=method)
    assert calls == [method]
    assert result["source"] is ordinary and arm["source"] is source
    assert before.completed_updates == before.training_examples == 0


@pytest.mark.parametrize(
    "fault",
    [
        "episode_mode",
        "updates",
        "updates_bool",
        "rows",
        "rows_bool",
        "policy",
        "completed",
        "completed_bool",
        "examples",
        "examples_bool",
        "method",
        "owner",
    ],
)
def test_should_refuse_non_catalogue_or_changed_source_before_ordinary_probe(monkeypatch, fault):
    source, before, arm = probe_source(monkeypatch, "pc")
    monkeypatch.setattr(route.online, "probe_arm", forbidden)
    if fault == "episode_mode":
        source._catalogue = None
    elif fault == "updates":
        source._catalogue_updates = 127
    elif fault == "updates_bool":
        source._catalogue_updates = True
    elif fault == "rows":
        source._catalogue_rows = 9
    elif fault == "rows_bool":
        source._catalogue_rows = True
    elif fault == "policy":
        before.policy = route.online._fixed_policy("neutral_cpc")
    elif fault == "completed":
        before.completed_updates = 1
    elif fault == "completed_bool":
        before.completed_updates = False
    elif fault == "examples":
        before.training_examples = 54
    elif fault == "examples_bool":
        before.training_examples = False
    elif fault == "method":
        arm["row"]["method"] = "public_invariant_bp"
    else:
        arm["source"] = object()
    with pytest.raises(ValueError):
        route.probe_arm(arm, method="pc")


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("describe", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_should_forward_explicit_cli_weighting_without_changing_original_joint_dispatch(
    monkeypatch, capsys, method, describe, weighted
):
    calls = []

    def selected(**settings):
        calls.append(settings)
        return dict(status="complete")

    monkeypatch.setattr(
        cli, "describe_minigrid_learnability_protocol", selected if describe else forbidden
    )
    monkeypatch.setattr(cli, "run_minigrid_learnability_pilot", forbidden if describe else selected)
    monkeypatch.setattr(
        sys,
        "argv",
        ["pilot", *CLI_FLAGS, method, *CLI_REQUIRED]
        + (["--weighted-catalogue"] if weighted else [])
        + (["--describe"] if describe else []),
    )
    cli.main()
    assert calls == [{**options(method), **({"weighted_catalogue": True} if weighted else {})}]
    assert json.loads(capsys.readouterr().out) == dict(status="complete")


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--scenario", "torch-minigrid-learnability"],
        [*CLI_FLAGS, "gated_cpc", *CLI_REQUIRED],
        *(
            [*CLI_FLAGS, "bp", *(flag for flag in CLI_REQUIRED if flag != missing)]
            for missing in CLI_REQUIRED
        ),
        [*CLI_FLAGS, "bp", *CLI_REQUIRED, "--weighted-bp"],
        [*CLI_FLAGS, "pc", *CLI_REQUIRED, "--weighted-pc"],
        # Weighted PC/neutral diagnostics are now supported; BP still refuses.
        [*CLI_FLAGS, "bp", *CLI_REQUIRED, "--pc-update-diagnostics"],
    ],
)
def test_should_refuse_invalid_cli_before_any_dispatch(monkeypatch, arguments):
    for name in (
        "describe_minigrid_learnability_protocol",
        "run_minigrid_learnability_pilot",
        "run_minigrid_pilot",
    ):
        monkeypatch.setattr(cli, name, forbidden)
    monkeypatch.setattr(sys, "argv", ["pilot", "--describe", "--weighted-catalogue", *arguments])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
