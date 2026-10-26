"""Explicit weighted PC/neutral routing with zero native updates or real games.

Admission binds complete TRAIN packets; scripted ports verify the existing fit
and disposable ordinary probes still run in order with their original guards.
"""

from copy import deepcopy
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
from src.adapters.torch_feature_learner import TORCH_PC_ENERGY_ID
from src.adapters.torch_pc_weighted import TORCH_INVARIANT_WEIGHTED_PC_ID
from test_minigrid_invariant_learnability import episode
from test_minigrid_learning_pilot import evaluator as evaluator
from test_minigrid_pc_qualification import environment
from test_torch_feature_rehearsal_pilot import forbidden


FLAGS: dict[str, Any] = dict(public_history=True, invariant_history=True, qualification_method="pc")
CLI_FLAGS = [
    "--scenario",
    "torch-minigrid-history-learnability",
    "--invariant-history",
    "--qualification-method",
]


def options(method="pc"):
    return {**FLAGS, "qualification_method": method}


def packets():
    raw = tuple(
        pilot._packet(episode(context, cue, "train", f"minigrid-source-{context}-{cue}"))
        for context in ("A", "B", "C_train")
        for cue in ("blue", "red")
    )
    return pilot._history_packets(pilot.shared._packets(raw))


@pytest.mark.parametrize("settings", [{}, FLAGS, dict(public_history=True)])
def test_should_preserve_original_descriptors_when_weighted_pc_is_false(settings):
    assert pilot.describe_protocol(**settings) == pilot.describe_protocol(
        **settings, weighted_pc=False
    )


@pytest.mark.parametrize("method", ["pc", "neutral_cpc"])
@pytest.mark.parametrize(
    "extra",
    [
        {},
        dict(activation_health=True),
        dict(distinct_probe=True, old_task_readouts=True, return_to_a=True),
    ],
)
def test_should_describe_separate_objectives_without_native_resources(monkeypatch, method, extra):
    for name in ("require_torch", "ProcessRssSampler", "MiniGridKeyDoor", "_history_learner"):
        monkeypatch.setattr(pilot, name, forbidden)
    monkeypatch.setattr(route, "TorchTrainWeightedLearner", forbidden)
    protocol = pilot.describe_protocol(**options(method), **extra, weighted_pc=True)
    ordinary = pilot.describe_protocol(**options(method), **extra)
    assert protocol["protocol_id"] == (
        ordinary["protocol_id"].removesuffix("_v1") + f"_weighted_{method}_v1"
    )
    assert protocol["methods"] == [f"public_invariant_{method}"]
    assert protocol["checkpoint_updates"] == ordinary["checkpoint_updates"]
    assert protocol["weighted_pc"]["benchmark_objective"] == TORCH_INVARIANT_WEIGHTED_PC_ID
    assert protocol["weighted_pc"]["probe_objective"] == TORCH_PC_ENERGY_ID
    assert protocol["probe"]["objective"] == TORCH_PC_ENERGY_ID
    assert protocol["probe"]["checkpoint_updates"] == [0, 4, 8, 16]
    if "return_phase" in protocol:
        assert protocol["return_phase"]["objective"] == TORCH_PC_ENERGY_ID
    assert (protocol["seed"], protocol["hidden_dim"], protocol["inference_steps"]) == (17, 8, 2)
    assert protocol["planned_work"]["weighted_benchmark_native_train_calls"] == 128
    assert protocol["planned_work"]["weighted_benchmark_training_rows"] == 1152
    assert protocol["planned_work"]["unweighted_probe_templates"] == 2
    assert protocol["planned_work"]["final_episodes"] == 0
    assert protocol["planned_work"]["replay_rows"] == 0
    assert protocol["metric_direction"] == ordinary["metric_direction"]
    json.dumps(protocol, allow_nan=False)


@pytest.mark.parametrize(
    "settings",
    [
        dict(weighted_pc=1),
        dict(weighted_pc=None),
        dict(weighted_pc=np.bool_(True)),
        dict(weighted_pc=True),
        dict(weighted_pc=True, public_history=True),
        dict(weighted_pc=True, public_history=True, invariant_history=True),
        dict(weighted_pc=True, **options("bp")),
        dict(weighted_pc=True, **options("gated_cpc")),
        dict(weighted_pc=True, **options("pc_relax64")),
        dict(weighted_pc=True, **FLAGS, pc_qualification=True),
        dict(weighted_pc=True, **FLAGS, memory_control=True),
        dict(weighted_pc=True, **FLAGS, weighted_bp=True),
        dict(weighted_pc=True, **FLAGS, joint_catalogue=True),
        dict(weighted_pc=True, **{**FLAGS, "public_history": 1}),
        dict(weighted_pc=True, **{**FLAGS, "invariant_history": 1}),
    ],
)
def test_should_refuse_unsupported_options_before_resources(monkeypatch, settings):
    for name in ("require_torch", "ProcessRssSampler", "MiniGridKeyDoor", "_history_learner"):
        monkeypatch.setattr(pilot, name, forbidden)
    monkeypatch.setattr(route, "TorchTrainWeightedLearner", forbidden)
    with pytest.raises(ValueError):
        pilot.describe_protocol(**settings)
    with pytest.raises(ValueError):
        pilot.run_pilot(**settings)


@pytest.mark.parametrize("describe", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("method", ["pc", "neutral_cpc"])
def test_should_forward_cli_weighting_only_when_explicit(
    monkeypatch, capsys, describe, weighted, method
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
        ["pilot", *CLI_FLAGS, method]
        + (["--weighted-pc"] if weighted else [])
        + (["--describe"] if describe else []),
    )
    cli.main()
    assert calls == [{**options(method), **({"weighted_pc": True} if weighted else {})}]
    assert json.loads(capsys.readouterr().out)["status"] == "complete"


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--scenario", "torch-minigrid-learnability"],
        ["--scenario", "torch-minigrid-history-learnability"],
        ["--scenario", "torch-minigrid-history-learnability", "--invariant-history"],
        [*CLI_FLAGS, "bp"],
        [*CLI_FLAGS, "gated_cpc"],
        [*CLI_FLAGS, "pc", "--pc-qualification"],
        [*CLI_FLAGS, "pc", "--weighted-bp"],
        [
            *CLI_FLAGS,
            "pc",
            "--joint-catalogue",
            "--distinct-probe",
            "--old-task-readouts",
            "--return-to-a",
        ],
    ],
)
def test_should_refuse_invalid_cli_before_dispatch(monkeypatch, arguments):
    for name in (
        "describe_minigrid_learnability_protocol",
        "run_minigrid_learnability_pilot",
        "run_minigrid_pilot",
    ):
        monkeypatch.setattr(cli, name, forbidden)
    monkeypatch.setattr(sys, "argv", ["pilot", "--describe", "--weighted-pc", *arguments])
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
        "current_geometry",
        "next_geometry",
        "row_ids",
        "context",
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
        elif fault in ("current_geometry", "next_geometry"):
            record = first.episode
            field = "observation" if fault == "current_geometry" else "next_observation"
            # A terminal next view has no adjacent row. Keep episode continuity
            # valid so this case reaches the route's public-view preflight.
            index = len(record.transitions) - 1 if field == "next_observation" else 0
            transitions = list(record.transitions)
            transitions[index] = replace(transitions[index], **{field: (0.0,) * 296})
            first = replace(
                first,
                episode=replace(
                    record,
                    transitions=tuple(transitions),
                ),
            )
        elif fault == "context":
            first = replace(first, context="B")
        else:
            first = replace(first, row_ids=(first.row_ids[0],) * 9)
        schedule = (first, *schedule[1:])
    monkeypatch.setattr(route, "TorchTrainWeightedLearner", forbidden)
    with pytest.raises(ValueError):
        route.factory(schedule, method="pc")("public_invariant_pc")


@pytest.mark.parametrize("method", ["pc", "public_invariant_bp", None, 1])
def test_should_refuse_mismatched_bound_method_before_native_constructor(monkeypatch, method):
    monkeypatch.setattr(route, "TorchTrainWeightedLearner", forbidden)
    with pytest.raises(ValueError):
        route.factory(packets(), method="neutral_cpc")(method)


@pytest.mark.parametrize("method", ["pc", "neutral_cpc"])
def test_should_bind_fixed_native_policy_and_own_complete_train_schedule(monkeypatch, method):
    schedule = packets()
    saved = pickle.dumps(schedule)
    calls = []

    def constructor(policy, episodes):
        calls.append((policy, episodes))
        return "owned"

    monkeypatch.setattr(route, "TorchTrainWeightedLearner", constructor)
    build = route.factory(schedule, method=method)
    assert build(f"public_invariant_{method}") == "owned"
    policy, episodes = calls[0]
    native = policy.native_policy
    assert (native.method, policy.feature_dim, policy.hidden_dim, policy.num_classes) == (
        method,
        296,
        8,
        7,
    )
    assert (
        native.seed,
        native.learning_rate,
        native.inference_steps,
        native.inference_learning_rate,
    ) == (17, 0.03, 2, 0.04)
    assert len(episodes) == 128 and all(len(record.transitions) == 9 for record in episodes)
    assert episodes is not schedule and pickle.dumps(schedule) == saved
    assert episodes == tuple(packet.episode for packet in schedule)
    assert all(
        owned is not packet.episode for owned, packet in zip(episodes, schedule, strict=True)
    )


def receipt_arm():
    return dict(
        row=dict(
            method="public_invariant_pc",
            stream_result=dict(
                evaluation_scope="training_only",
                completed_updates=128,
                training_examples=1152,
                applied_updates=[
                    dict(
                        update_number=index,
                        diagnostic=dict(definition=TORCH_INVARIANT_WEIGHTED_PC_ID, value=0.25),
                    )
                    for index in range(1, 129)
                ],
            ),
        )
    )


def test_should_accept_only_dynamic_pc_weighted_receipts():
    route.check_receipts(receipt_arm(), method="pc")


@pytest.mark.parametrize(
    "fault",
    [
        "updates",
        "rows",
        "scope",
        "short",
        "counter",
        "definition",
        "nonfinite",
        "replaced_receipt",
        "method",
    ],
)
def test_should_refuse_corrupted_paid_pc_evidence(fault):
    arm = receipt_arm()
    result = arm["row"]["stream_result"]
    receipt = result["applied_updates"][0]
    if fault == "updates":
        result["completed_updates"] = 127
    elif fault == "rows":
        result["training_examples"] = 1151
    elif fault == "scope":
        result["evaluation_scope"] = "final"
    elif fault == "short":
        result["applied_updates"].pop()
    elif fault == "counter":
        receipt["update_number"] = True
    elif fault == "definition":
        receipt["diagnostic"]["definition"] = TORCH_PC_ENERGY_ID
    elif fault == "nonfinite":
        receipt["diagnostic"]["value"] = float("nan")
    elif fault == "replaced_receipt":
        result["applied_updates"][0] = None
    else:
        arm["row"]["method"] = "public_invariant_neutral_cpc"
    with pytest.raises(ValueError):
        route.check_receipts(arm, method="pc")


@pytest.mark.parametrize("method", ["pc", "neutral_cpc"])
def test_should_keep_original_fit_and_two_unweighted_disposable_probes(
    monkeypatch, evaluator, method
):
    case = environment(monkeypatch, evaluator, method=f"public_invariant_{method}")
    events = []
    original = pilot._history_learner

    def factory(schedule, *, method):
        events.append(("factory", method))
        assert len(schedule) == 128
        return original

    def receipts(arm, *, method):
        arm["guard"]()
        events.append(("receipts", method))

    def probe_arm(arm, *, method):
        events.append(("probe", method))
        return dict(arm)

    monkeypatch.setattr(route, "factory", factory)
    monkeypatch.setattr(route, "check_receipts", receipts)
    monkeypatch.setattr(route, "probe_arm", probe_arm)
    report = pilot.run_pilot(**options(method), weighted_pc=True)
    assert report["status"] == "complete", report["reason"]
    assert len(case.work["fit"]) == 5 and len(case.work["probes"]) == 2
    assert all(selected == method for _, selected in events)
    assert [kind for kind, _ in events].count("factory") == 1
    assert [kind for kind, _ in events].count("probe") == 2
    assert any(kind == "receipts" for kind, _ in events)
    assert [call["state"].completed_updates for call in case.work["probes"]] == [0, 128]
    assert all(call["source_before"] == call["source_after"] for call in case.work["probes"])
    assert all(call["inputs_before"] == call["inputs_after"] for call in case.work["probes"])
    assert all(call["report"]["completed_updates"] == 16 for call in case.work["probes"])
    assert report["methods"][0]["source_unchanged"] is True
    assert report["methods"][0]["actor_unchanged"] is True
    assert report["scientific_advantage_verified"] is False
    frozen = deepcopy(case.work["arms"][0]["source"].snapshot_state())
    case.work["arms"][0]["guard"]()
    assert pickle.dumps(frozen) == pickle.dumps(case.work["arms"][0]["source"].snapshot_state())
