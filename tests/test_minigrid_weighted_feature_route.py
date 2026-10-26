"""Opt-in weighted TRAIN routing and isolated ordinary fresh-task probes.

One synthetic native128-update stream and one ordinary16-update probe are the
only paid model work in this module. Admission and CLI cases construct no models.
"""

from dataclasses import replace
import json
import pickle
import sys
from typing import Any

import numpy as np
import pytest

import scripts.run_coupled_attribution as cli
from src.adapters import minigrid_learnability as pilot
from src.adapters import minigrid_weighted_feature_route as route
from src.adapters.invariant_torch_feature_learner import InvariantTorchFeatureLearner
from src.adapters.torch_feature_learner import TORCH_CE_ID
from src.adapters.torch_feature_weighted import TORCH_INVARIANT_WEIGHTED_CE_ID
from src.adapters.torch_train_weighted_learner import TorchTrainWeightedLearner
from src.app import toy_execution_budget as budgets
from src.shared.torch_runtime import require_torch
from test_minigrid_invariant_learnability import episode, probe_data
from test_minigrid_learning_pilot import evaluator as evaluator
from test_minigrid_pc_qualification import environment
from test_torch_feature_learner import assert_tree
from test_torch_feature_rehearsal_pilot import Sampler, forbidden


METHOD = "public_invariant_bp"
FLAGS: dict[str, Any] = dict(public_history=True, invariant_history=True, qualification_method="bp")
CLI_FLAGS = [
    "--scenario",
    "torch-minigrid-history-learnability",
    "--invariant-history",
    "--qualification-method",
    "bp",
]


def packets():
    raw = tuple(
        pilot._packet(episode(context, cue, "train", f"minigrid-source-{context}-{cue}"))
        for context in ("A", "B", "C_train")
        for cue in ("blue", "red")
    )
    return pilot._history_packets(pilot.shared._packets(raw))


@pytest.mark.parametrize("options", [{}, FLAGS, dict(public_history=True)])
def test_should_leave_original_descriptors_identical_when_weighting_is_false(options):
    assert pilot.describe_protocol(**options) == pilot.describe_protocol(
        **options, weighted_bp=False
    )


@pytest.mark.parametrize(
    "extra",
    [
        {},
        dict(distinct_probe=True),
        dict(activation_health=True),
        dict(prior_gradient_geometry=True),
        dict(acquisition_only=True),
        dict(distinct_probe=True, old_task_readouts=True, return_to_a=True),
    ],
)
def test_should_describe_explicit_objectives_without_native_resources(monkeypatch, extra):
    for name in ("require_torch", "ProcessRssSampler", "MiniGridKeyDoor", "_history_learner"):
        monkeypatch.setattr(pilot, name, forbidden)
    monkeypatch.setattr(route, "TorchTrainWeightedLearner", forbidden)
    protocol = pilot.describe_protocol(**FLAGS, **extra, weighted_bp=True)
    ordinary = pilot.describe_protocol(**FLAGS, **extra)
    assert (
        protocol["protocol_id"] == ordinary["protocol_id"].removesuffix("_v1") + "_weighted_bp_v1"
    )
    assert protocol["methods"] == [METHOD]
    assert protocol["checkpoint_updates"] == ordinary["checkpoint_updates"]
    assert protocol["probe"]["checkpoint_updates"] == [0, 4, 8, 16]
    assert protocol["weighted_bp"]["benchmark_objective"] == TORCH_INVARIANT_WEIGHTED_CE_ID
    assert protocol["weighted_bp"]["probe_objective"] == TORCH_CE_ID
    assert protocol["planned_work"]["weighted_benchmark_native_train_calls"] == 128
    assert protocol["planned_work"]["unweighted_probe_templates"] == 2
    assert protocol["planned_work"]["final_episodes"] == 0
    assert protocol["planned_work"]["replay_rows"] == 0


@pytest.mark.parametrize(
    "options",
    [
        dict(weighted_bp=1),
        dict(weighted_bp=None),
        dict(weighted_bp=np.bool_(True)),
        dict(weighted_bp=True),
        dict(weighted_bp=True, public_history=True),
        dict(weighted_bp=True, public_history=True, invariant_history=True),
        dict(weighted_bp=True, **{**FLAGS, "qualification_method": "pc"}),
        dict(weighted_bp=True, **{**FLAGS, "qualification_method": "neutral_cpc"}),
        dict(weighted_bp=True, **{**FLAGS, "qualification_method": "gated_cpc"}),
        dict(weighted_bp=True, **FLAGS, pc_qualification=True),
        dict(weighted_bp=True, **FLAGS, memory_control=True),
    ],
)
def test_should_refuse_unsupported_options_before_any_resources(monkeypatch, options):
    for name in ("require_torch", "ProcessRssSampler", "MiniGridKeyDoor", "_history_learner"):
        monkeypatch.setattr(pilot, name, forbidden)
    monkeypatch.setattr(route, "TorchTrainWeightedLearner", forbidden)
    with pytest.raises(ValueError):
        pilot.describe_protocol(**options)
    with pytest.raises(ValueError):
        pilot.run_pilot(**options)


@pytest.mark.parametrize("describe", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_should_forward_cli_weighting_only_when_explicit(monkeypatch, capsys, describe, weighted):
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
        + (["--weighted-bp"] if weighted else [])
        + (["--describe"] if describe else []),
    )
    cli.main()
    assert calls == [{**FLAGS, **({"weighted_bp": True} if weighted else {})}]
    assert json.loads(capsys.readouterr().out)["status"] == "complete"


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--scenario", "torch-minigrid-learnability"],
        ["--scenario", "torch-minigrid-history-learnability"],
        ["--scenario", "torch-minigrid-history-learnability", "--invariant-history"],
        [*CLI_FLAGS[:-1], "pc"],
        [*CLI_FLAGS, "--pc-qualification"],
        [*CLI_FLAGS, "--memory-control"],
    ],
)
def test_should_refuse_invalid_cli_before_dispatch(monkeypatch, arguments):
    for name in (
        "describe_minigrid_learnability_protocol",
        "run_minigrid_learnability_pilot",
        "run_minigrid_pilot",
    ):
        monkeypatch.setattr(cli, name, forbidden)
    monkeypatch.setattr(sys, "argv", ["pilot", "--describe", "--weighted-bp", *arguments])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


@pytest.mark.parametrize(
    "fault",
    [
        "list",
        "short",
        "missing_episode",
        "role",
        "classes",
        "features",
        "targets",
        "dtype",
        "episode_role",
        "episode_geometry",
        "row_ids",
    ],
)
def test_should_refuse_bad_schedule_before_native_constructor(monkeypatch, fault):
    schedule = packets()
    first = schedule[0]
    if fault == "list":
        schedule = list(schedule)
    elif fault == "short":
        schedule = schedule[:-1]
    else:
        if fault == "missing_episode":
            first = replace(first, episode=None)
        elif fault == "role":
            first = replace(first, role="probe_train")
        elif fault == "classes":
            first = replace(first, num_classes=8)
        elif fault == "features":
            features = first.features.copy()
            features[0, 0] += 1
            first = replace(first, features=features)
        elif fault == "targets":
            targets = first.targets.copy()
            targets[0, 0] = (targets[0, 0] + 1) % 7
            first = replace(first, targets=targets)
        elif fault == "dtype":
            first = replace(first, features=first.features.astype(np.float32))
        elif fault == "episode_role":
            first = replace(first, episode=replace(first.episode, role="probe_test"))
        elif fault == "episode_geometry":
            record = first.episode
            first = replace(
                first,
                episode=replace(
                    record,
                    transitions=(
                        replace(record.transitions[0], observation=(0.0,) * 296),
                        *record.transitions[1:],
                    ),
                ),
            )
        else:
            first = replace(first, row_ids=(first.row_ids[0],) * 9)
        schedule = (first, *schedule[1:])
    monkeypatch.setattr(route, "TorchTrainWeightedLearner", forbidden)
    with pytest.raises(ValueError):
        route.factory(schedule)(METHOD)


@pytest.mark.parametrize("method", ["bp", "public_invariant_pc", None, 1])
def test_should_refuse_other_method_before_native_constructor(monkeypatch, method):
    monkeypatch.setattr(route, "TorchTrainWeightedLearner", forbidden)
    with pytest.raises(ValueError):
        route.factory(packets())(method)


def test_should_bind_fixed_bp_policy_and_own_the_complete_schedule(monkeypatch):
    schedule = packets()
    saved = pickle.dumps(schedule)
    calls = []

    def constructor(policy, episodes):
        calls.append((policy, episodes))
        return "owned"

    monkeypatch.setattr(route, "TorchTrainWeightedLearner", constructor)
    factory = route.factory(schedule)
    assert factory(METHOD) == "owned"
    policy, episodes = calls[0]
    assert (
        policy.native_policy.method,
        policy.feature_dim,
        policy.hidden_dim,
        policy.num_classes,
    ) == ("bp", 296, 8, 7)
    assert policy.native_policy.seed == 17
    assert len(episodes) == 128 and all(len(record.transitions) == 9 for record in episodes)
    assert episodes is not schedule
    assert pickle.dumps(schedule) == saved
    assert episodes == tuple(packet.episode for packet in schedule)


def test_should_route_whole_pilot_without_replacing_existing_fit_probe_or_guards(
    monkeypatch, evaluator
):
    case = environment(monkeypatch, evaluator, method=METHOD)
    events = []
    original_factory = pilot._history_learner
    monkeypatch.setattr(route, "factory", lambda schedule: original_factory)

    def receipt_check(arm):
        arm["guard"]()
        events.append(("receipts", arm))

    def probe_arm(arm):
        events.append(("probe", arm))
        return dict(arm)

    monkeypatch.setattr(route, "check_receipts", receipt_check)
    monkeypatch.setattr(route, "probe_arm", probe_arm)
    report = pilot.run_pilot(**FLAGS, weighted_bp=True)
    assert report["status"] == "complete", report["reason"]
    assert len(case.work["fit"]) == 5 and len(case.work["probes"]) == 2
    assert any(kind == "receipts" for kind, _ in events)
    assert [kind for kind, _ in events].count("probe") == 2
    assert all(arm is case.work["arms"][0] for _, arm in events)
    assert report["methods"][0]["source_unchanged"] is True
    assert report["methods"][0]["actor_unchanged"] is True
    assert report["scientific_advantage_verified"] is False


def test_should_execute_weighted_delayed_arm_then_an_isolated_unweighted_probe(monkeypatch):
    torch = require_torch()
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    clock = lambda: 1.0
    monkeypatch.setattr(pilot, "monotonic", clock)
    monkeypatch.setattr(pilot.shared, "monotonic", clock)
    monkeypatch.setattr(budgets, "ProcessRssSampler", Sampler)
    monkeypatch.setattr(pilot, "MiniGridKeyDoor", forbidden)
    monkeypatch.setattr(pilot, "CueKeyDoorTeacher", forbidden)
    schedule = packets()
    saved_inputs = pickle.dumps(schedule)
    row = pilot._new_row(METHOD)
    arm = pilot.shared._train_arm(
        row, schedule, Sampler(), 100.0, lambda: None, learner_factory=route.factory(schedule)
    )
    arm["guard"]()
    route.check_receipts(arm)
    source = arm["source"]
    assert type(source) is TorchTrainWeightedLearner
    initial = source.snapshot_state()
    assert initial.completed_updates == initial.training_examples == 0
    assert row["source_unchanged"] is row["actor_unchanged"] is True
    assert arm["budget"].updates_completed == 128
    assert arm["budget"].replay_examples_completed == 0
    assert [point for point, _ in arm["states"]] == [0, 32, 64, 96, 128]
    receipts = row["stream_result"]["applied_updates"]
    assert len(receipts) == 128
    assert all(
        receipt["diagnostic"]["definition"] == TORCH_INVARIANT_WEIGHTED_CE_ID
        for receipt in receipts
    )
    state = arm["states"][-1][1]
    assert (state.completed_updates, state.training_examples) == (128, 1152)
    probe = route.probe_arm(arm)
    assert probe is not arm and probe["row"] is arm["row"]
    assert type(probe["source"]) is InvariantTorchFeatureLearner
    assert probe["source"] is not source
    assert_tree(torch, probe["source"].snapshot_state(), initial)
    assert_tree(torch, source.snapshot_state(), initial)
    calls = []
    original = InvariantTorchFeatureLearner.train_batch

    def train(owner, features, targets):
        calls.append(owner)
        return original(owner, features, targets)

    monkeypatch.setattr(InvariantTorchFeatureLearner, "train_batch", train)
    slot = row["probes"][1]
    completed = pilot._run_probe(
        slot,
        probe,
        state,
        probe_data(),
        sampler=Sampler(),
        deadline=100.0,
        check_boundary=arm["guard"],
    )
    completed()
    arm["guard"]()
    route.check_receipts(arm)
    assert slot["status"] == slot["report"]["status"] == "complete"
    assert slot["charged_updates"] == slot["report"]["completed_updates"] == 16
    assert slot["report"]["completed_training_examples"] == 144
    assert slot["native_updates_before"] == 128
    assert len(calls) == 16 and all(
        owner is not source and owner is not probe["source"] for owner in calls
    )
    assert_tree(torch, source.snapshot_state(), initial)
    assert_tree(torch, probe["source"].snapshot_state(), initial)
    assert_tree(torch, arm["stream"]._runtime.actor.snapshot_state().state, initial)
    assert arm["budget"].updates_completed == 128
    assert pickle.dumps(schedule) == saved_inputs

    # The paid original proof must still reject replacement or corruption.
    for key, value in (("completed_updates", 127), ("training_examples", 1151)):
        result = row["stream_result"]
        saved = result[key]
        result[key] = value
        with pytest.raises(ValueError):
            route.check_receipts(arm)
        result[key] = saved
    diagnostic = receipts[0]["diagnostic"]
    saved = diagnostic["definition"]
    diagnostic["definition"] = TORCH_CE_ID
    with pytest.raises(ValueError):
        route.check_receipts(arm)
    diagnostic["definition"] = saved
    arm["guard"]()
    route.check_receipts(arm)
