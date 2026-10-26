"""Public joint-catalogue admission and composition without native learner work.

Descriptors and CLI dispatch use forbidden resource constructors. Synthetic
complete TRAIN histories exercise the real catalogue helper, not physics, SGD,
checkpoint probes or a substitute orchestration loop.
"""

from dataclasses import replace
import json
import pickle
import sys
from typing import Any

import numpy as np
import pytest

import scripts.run_coupled_attribution as cli
from src.adapters import minigrid_joint_catalogue_route as route
from src.adapters import minigrid_learnability as pilot
from test_minigrid_train_catalogue import episode


METHODS = ("bp", "pc", "neutral_cpc", "gated_cpc")
FLAGS: dict[str, Any] = dict(
    public_history=True,
    invariant_history=True,
    qualification_method="bp",
    distinct_probe=True,
    old_task_readouts=True,
    return_to_a=True,
)
CLI_FLAGS = [
    "--scenario",
    "torch-minigrid-history-learnability",
    "--invariant-history",
    "--qualification-method",
    "bp",
    "--distinct-probe",
    "--old-task-readouts",
    "--return-to-a",
]
EXCLUDED = (
    "pc_qualification",
    "acquisition_only",
    "weighted_bp",
    "memory_control",
    "activation_health",
    "gradient_geometry",
    "prior_gradient_geometry",
    "probe_rollouts",
    "deferred_probe_rollouts",
)


def forbidden(*args, **kwargs):
    raise AssertionError("admission/description must not construct runtime resources")


@pytest.fixture
def no_resources(monkeypatch):
    for name in ("require_torch", "ProcessRssSampler", "MiniGridKeyDoor", "_history_learner"):
        monkeypatch.setattr(pilot, name, forbidden)
    monkeypatch.setattr(pilot.shared, "_train_arm", forbidden)
    monkeypatch.setattr(pilot.shared, "_collect_sources", forbidden)


def packets():
    return tuple(
        pilot._packet(
            episode(
                f"minigrid-source-{context}-{cue}",
                context=context,
                cue=2 if cue == "blue" else 0,
                actions=(0, 1, 2, 2, 2, 2, 3, 5, 2),
            )
        )
        for context in ("A", "B", "C_train")
        for cue in ("blue", "red")
    )


@pytest.mark.parametrize("options", [{}, dict(public_history=True), FLAGS])
def test_should_leave_original_protocol_identical_when_catalogue_is_false(no_resources, options):
    assert pilot.describe_protocol(**options) == pilot.describe_protocol(
        **options, joint_catalogue=False
    )


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("margins", [False, True])
def test_should_describe_selected_unchanged_method_and_explicit_joint_dose(
    no_resources, method, margins
):
    protocol = pilot.describe_protocol(
        **{**FLAGS, "qualification_method": method},
        joint_catalogue=True,
        train_action_margins=margins,
    )
    assert protocol["methods"] == [f"public_invariant_{method}"]
    assert protocol["seed"] == 17 and protocol["hidden_dim"] == 8
    assert protocol["learning_rate"] == 0.03 and protocol["momentum"] == 0.9
    assert protocol["inference_steps"] == 2 and protocol["inference_learning_rate"] == 0.04
    assert protocol["checkpoint_updates"] == [0, 32, 64, 96, 128]
    assert protocol["primary_comparisons"] == []
    assert "differences" not in protocol
    assert protocol["planned_work"]["benchmark_native_train_calls"] == 128
    assert protocol["planned_work"]["benchmark_training_rows"] == 128 * 54
    assert protocol["planned_work"]["probe_native_train_calls"] == 32
    assert protocol["planned_work"]["probe_training_rows"] == 32 * 9
    assert protocol["planned_work"]["return_native_train_calls"] == 32
    assert protocol["planned_work"]["return_training_rows"] == 32 * 9
    assert protocol["planned_work"]["native_train_calls"] == 192
    assert protocol["planned_work"]["training_rows"] == 6912 + 288 + 288
    assert protocol["planned_work"]["known_raw_score_forwards"] == 30
    assert protocol["planned_work"]["known_raw_score_rows"] == 270
    assert protocol["planned_work"]["final_episodes"] == 0
    assert protocol["planned_work"]["replay_rows"] == 0
    # Every wake repeats precisely these six sources. The support set therefore
    # needs each complete episode once, while paid benchmark exposure stays6912.
    support = protocol["probe_support"]
    assert support["training_packets"] == 6 and support["training_rows"] == 54
    assert support["probe_packets"] == 18 and support["probe_rows"] == 162
    assert support["history_rows"] == support["canonical_projection_rows"] == 216
    assert support["canonical_projected_array_equivalent_bytes"] == 216 * 2368
    assert support["training_scope"] and support["exposure_binding"]
    assert protocol["probe"]["source_stages"] == ["initial", "post_joint128"]
    assert protocol["joint_catalogue"]["rows_per_update"] == 54
    assert protocol["joint_catalogue"]["training_row_exposures"] == 6912


@pytest.mark.parametrize("flag", [None, 0, 1, "true", np.bool_(True)])
def test_should_refuse_non_native_catalogue_flags_before_resources(no_resources, flag):
    with pytest.raises(ValueError):
        pilot.describe_protocol(**FLAGS, joint_catalogue=flag)
    with pytest.raises(ValueError):
        pilot.run_pilot(**FLAGS, joint_catalogue=flag)


@pytest.mark.parametrize(
    "changed",
    [
        dict(public_history=False),
        dict(invariant_history=False),
        dict(qualification_method=None),
        dict(qualification_method="recurrent_bc"),
        dict(distinct_probe=False),
        dict(old_task_readouts=False),
        dict(return_to_a=False),
        *({key: True} for key in EXCLUDED),
    ],
)
def test_should_refuse_incomplete_or_incompatible_joint_options_before_resources(
    no_resources, changed
):
    options = {**FLAGS, **changed, "joint_catalogue": True}
    with pytest.raises(ValueError):
        pilot.describe_protocol(**options)
    with pytest.raises(ValueError):
        pilot.run_pilot(**options)


def test_should_compose_all_six_sources_in_original_order_with_owned_repeated_arrays():
    sources = packets()
    saved = pickle.dumps(sources)
    plan = route.compose(sources)
    assert len(plan.training) == 128 and plan.source_rows == 54
    assert plan.row_exposures == 6912
    assert plan.input_array_bytes == 128 * 54 * 297 * 8
    assert plan.source_template_array_bytes == 54 * 297 * 8
    assert plan.validation_projection_rows == 108
    assert plan.validation_projected_array_equivalent_bytes == 108 * 296 * 8
    assert [span.episode_id for span in plan.source_spans] == [
        packet.episode.episode_id for packet in sources
    ]
    assert [(span.start, span.stop) for span in plan.source_spans] == [
        (9 * index, 9 * (index + 1)) for index in range(6)
    ]
    wanted_features = np.concatenate([packet.features for packet in sources])
    wanted_targets = np.concatenate([packet.targets for packet in sources])
    for batch in plan.training:
        assert batch.role == "train" and batch.context == "train-catalogue"
        assert batch.episode is None and batch.num_classes == 7
        assert batch.features.shape == (54, 296) and batch.targets.shape == (54, 1)
        np.testing.assert_array_equal(batch.features, wanted_features)
        np.testing.assert_array_equal(batch.targets, wanted_targets)
    first, second = plan.training[:2]
    assert not np.shares_memory(first.features, second.features)
    assert not np.shares_memory(first.targets, second.targets)
    first.features.fill(0)
    first.targets.fill(6)
    np.testing.assert_array_equal(second.features, wanted_features)
    np.testing.assert_array_equal(second.targets, wanted_targets)
    assert pickle.dumps(sources) == saved


@pytest.mark.parametrize(
    "fault",
    ["list", "short", "order", "identity", "context", "missing_episode", "role", "rows", "targets"],
)
def test_should_refuse_source_catalogue_substitutions_before_composition(fault):
    sources: Any = packets()
    if fault == "list":
        sources = list(sources)
    elif fault == "short":
        sources = sources[:-1]
    elif fault == "order":
        sources = (sources[1], sources[0], *sources[2:])
    else:
        first = sources[0]
        record = first.episode
        if fault == "identity":
            first = replace(first, episode=replace(record, episode_id="other-source-blue"))
        elif fault == "context":
            first = replace(first, context="B")
        elif fault == "missing_episode":
            first = replace(first, episode=None)
        elif fault == "role":
            first = replace(first, role="probe_train")
        elif fault == "rows":
            first = replace(first, features=first.features[:-1], targets=first.targets[:-1])
        else:
            targets = first.targets.copy()
            targets[-1, 0] = (targets[-1, 0] + 1) % 7
            first = replace(first, targets=targets)
        sources = (first, *sources[1:])
    with pytest.raises(ValueError):
        route.compose(sources)


@pytest.mark.parametrize("describe", [False, True])
@pytest.mark.parametrize("joint", [False, True])
def test_should_forward_cli_joint_flag_only_when_selected(monkeypatch, capsys, describe, joint):
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
        + (["--joint-catalogue"] if joint else [])
        + (["--describe"] if describe else []),
    )
    cli.main()
    assert calls == [{**FLAGS, **({"joint_catalogue": True} if joint else {})}]
    assert json.loads(capsys.readouterr().out) == dict(status="complete")


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--scenario", "torch-minigrid-learnability"],
        ["--scenario", "torch-minigrid-history-learnability"],
        CLI_FLAGS[:-1],
        [flag for flag in CLI_FLAGS if flag != "--distinct-probe"],
        [flag for flag in CLI_FLAGS if flag != "--old-task-readouts"],
        [*CLI_FLAGS, "--acquisition-only"],
        [*CLI_FLAGS, "--weighted-bp"],
        [*CLI_FLAGS, "--memory-control"],
        [*CLI_FLAGS, "--native-activation-health"],
        [*CLI_FLAGS, "--cue-gradient-geometry"],
        [*CLI_FLAGS, "--prior-gradient-geometry"],
        [*CLI_FLAGS, "--probe-rollouts"],
        [*CLI_FLAGS, "--deferred-probe-rollouts"],
    ],
)
def test_should_refuse_invalid_cli_before_any_dispatch(monkeypatch, arguments):
    for name in (
        "describe_minigrid_learnability_protocol",
        "run_minigrid_learnability_pilot",
        "run_minigrid_pilot",
    ):
        monkeypatch.setattr(cli, name, forbidden)
    monkeypatch.setattr(sys, "argv", ["pilot", "--describe", "--joint-catalogue", *arguments])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
