"""Joint-catalogue known-A games using accounting fakes, without model or physics work.

The complete native snapshot copier has separate native coverage. Its fake here
lets the real recipe and row counter enforce admission before target-free games.
"""

from copy import deepcopy
from dataclasses import replace
import json
import pickle
import sys
from typing import Any, cast

import pytest

import scripts.run_coupled_attribution as cli
from src.adapters import minigrid_learnability as pilot
from src.adapters import minigrid_learning_pilot as shared
from src.adapters import minigrid_policy_evaluation as evaluation
from src.adapters import minigrid_relaxation_control as relaxation
from src.adapters import minigrid_train_rollout as rollout
from src.adapters.invariant_torch_feature_learner import InvariantTorchFeaturePolicy
from src.adapters.torch_feature_learner import TorchFeaturePolicy
from test_minigrid_joint_catalogue_route import CLI_FLAGS, FLAGS, forbidden
from test_minigrid_joint_catalogue_route import no_resources as no_resources
from test_minigrid_train_a_evaluation import case as case
from test_minigrid_train_a_rollout import live_report


METHODS = ("bp", "pc", "neutral_cpc", "gated_cpc", "pc_relax64", "neutral_cpc_relax64")


def joint_state(case, monkeypatch, method="pc_relax64"):
    monkeypatch.setattr(shared, "copy_invariant_torch_feature_snapshot", deepcopy)
    base = method.removesuffix("_relax64")
    parts = case.build(base)
    policy = (
        relaxation.policy(f"public_invariant_{method}")
        if method in relaxation.METHODS
        else InvariantTorchFeaturePolicy(TorchFeaturePolicy(base, 296, 8, 17, num_classes=7))
    )
    parts.source.policy = parts.source.state.policy = parts.state.policy = policy
    parts.state.training_examples = 6912
    return parts


def evaluate(parts, method="pc_relax64", **options):
    return evaluation.evaluate_neural_representation_checkpoint(
        parts.source,
        parts.state,
        updates=options.pop("updates", 128),
        budget=parts.budget,
        check_boundary=lambda: None,
        known_a_only=options.pop("known_a_only", True),
        catalogue_method=method if method is None else f"public_invariant_{method}",
        **options,
    )


@pytest.mark.parametrize("method", METHODS)
def test_should_read_two_target_free_a_games_from_exact_joint_endpoint(case, monkeypatch, method):
    parts = joint_state(case, monkeypatch, method)
    monkeypatch.setattr(evaluation, "predefined_probe_layouts", forbidden)
    before = pickle.dumps((parts.source.snapshot_state(), parts.state))
    report = evaluate(parts, method)
    assert report["status"] == "complete" and report["state_unchanged"] is True
    assert report["method"] == f"public_invariant_{method}"
    assert report["native_method"] == method.removesuffix("_relax64")
    assert report["updates"] == 128 and report["native_training_examples"] == 6912
    assert report["role"] == "knownTRAIN_A_only" and report["scores"] == []
    assert [(slot["context"], slot["cue"], slot["source_role"]) for slot in report["episodes"]] == [
        ("A", "blue", "train"),
        ("A", "red", "train"),
    ]
    assert report["work"]["environment_resets_completed"] == 2
    assert report["work"]["live_prediction_calls_returned"] == 4
    assert report["work"]["environment_steps_returned"] == 4
    assert report["work"]["raw_score_forwards_completed"] == 0
    for slot in report["episodes"]:
        initial = tuple(slot["predictions"][0]["observation"])
        for decision in slot["predictions"]:
            assert tuple(decision["features"]) == initial + tuple(decision["observation"])
            assert decision["action"] == 0
            assert decision["probabilities"] == [0.5, 0.5, 0, 0, 0, 0, 0]
    assert parts.budget.updates_completed == parts.budget.replay_examples_completed == 0
    assert pickle.dumps((parts.source.snapshot_state(), parts.state)) == before
    assert all(environment.closed for environment in case.work["environments"])


@pytest.mark.parametrize(
    "method",
    ["pc", "public_invariant_recurrent_bc", "public_invariant_gated_cpc_relax64", "", 1, []],
)
def test_should_refuse_undeclared_catalogue_recipe_before_ports(case, monkeypatch, method):
    parts = joint_state(case, monkeypatch)
    monkeypatch.setattr(evaluation, "copy_invariant_torch_feature_snapshot", forbidden)
    monkeypatch.setattr(evaluation, "MiniGridKeyDoor", forbidden)
    with pytest.raises(ValueError):
        evaluation.evaluate_neural_representation_checkpoint(
            parts.source,
            parts.state,
            updates=128,
            budget=parts.budget,
            check_boundary=forbidden,
            known_a_only=True,
            catalogue_method=method,
        )


@pytest.mark.parametrize(
    "change", [dict(known_a_only=False), dict(updates=0), dict(updates=32), dict(updates=True)]
)
def test_should_refuse_joint_selection_without_exact_known_a_endpoint_before_ports(
    case, monkeypatch, change
):
    parts = joint_state(case, monkeypatch)
    monkeypatch.setattr(evaluation, "copy_invariant_torch_feature_snapshot", forbidden)
    with pytest.raises(ValueError):
        evaluate(parts, **change)


@pytest.mark.parametrize("rows", [1152, 6911, 6913, 6912.0, True])
def test_should_refuse_relabelled_joint_row_counts_before_game_admission(case, monkeypatch, rows):
    parts = joint_state(case, monkeypatch)
    parts.state.training_examples = rows
    monkeypatch.setattr(parts.source, "fork", forbidden)
    monkeypatch.setattr(evaluation, "MiniGridKeyDoor", forbidden)
    with pytest.raises(ValueError):
        evaluate(parts)


@pytest.mark.parametrize("fault", ["s2", "other_algorithm", "rate"])
def test_should_refuse_changed_fixed_recipe_before_game_admission(case, monkeypatch, fault):
    parts = joint_state(case, monkeypatch)
    native = parts.state.policy.native_policy
    changed = replace(
        native,
        **cast(
            dict[str, Any],
            {
                "s2": {"inference_steps": 2},
                "other_algorithm": {"method": "neutral_cpc"},
                "rate": {"learning_rate": 0.031},
            }[fault],
        ),
    )
    parts.state.policy = InvariantTorchFeaturePolicy(changed)
    monkeypatch.setattr(evaluation, "MiniGridKeyDoor", forbidden)
    with pytest.raises(ValueError):
        evaluate(parts)


@pytest.mark.parametrize("source_updates", [32, 128])
def test_should_refuse_a_noninitial_source_before_fork_or_reset(case, monkeypatch, source_updates):
    parts = joint_state(case, monkeypatch)
    parts.source.state.completed_updates = source_updates
    parts.source.state.training_examples = 54 * source_updates
    forks = []
    original_fork = parts.source.fork

    def fork():
        forks.append(True)
        return original_fork()

    monkeypatch.setattr(parts.source, "fork", fork)
    monkeypatch.setattr(evaluation, "MiniGridKeyDoor", forbidden)
    with pytest.raises(evaluation.MiniGridEvaluationFailed) as caught:
        evaluate(parts)
    assert caught.value.report["work"]["live_prediction_calls_attempted"] == 0
    assert forks == []
    assert not case.work.get("environments")


@pytest.mark.parametrize("method", ["pc", "pc_relax64"])
def test_should_keep_default_evaluator_closed_to_joint_or_s64_state(case, monkeypatch, method):
    parts = joint_state(case, monkeypatch, method)
    with pytest.raises(ValueError):
        evaluate(parts, None)


@pytest.mark.parametrize("method", METHODS)
def test_should_add_live_plan_without_changing_joint_training_or_disposable_dose(
    no_resources, method
):
    options = {**FLAGS, "qualification_method": method, "joint_catalogue": True}
    old = pilot.describe_protocol(**options)
    selected = pilot.describe_protocol(**options, train_a_rollout=True)
    for key in (
        "seed",
        "hidden_dim",
        "learning_rate",
        "momentum",
        "inference_steps",
        "cpc_configs",
        "checkpoint_updates",
        "joint_catalogue",
    ):
        assert selected[key] == old[key]
    for key in (
        "native_train_calls",
        "training_rows",
        "score_forwards",
        "score_rows",
        "final_episodes",
    ):
        assert selected["planned_work"][key] == old["planned_work"][key]
    assert selected["planned_work"]["native_train_calls"] == 192
    assert selected["planned_work"]["training_rows"] == 7488
    assert selected["planned_work"]["learner_live_episodes"] == 2
    assert selected["planned_work"]["maximum_live_prediction_forwards"] == 32
    assert selected["planned_work"]["maximum_total_prediction_forwards"] == 126
    assert selected["train_a_rollout"]["cues"] == ["blue", "red"]
    assert selected["train_a_rollout"]["role"] == "train"


@pytest.mark.parametrize("method", ["bp", "pc_relax64"])
@pytest.mark.parametrize("describe", [False, True])
def test_should_forward_explicit_joint_live_cli_without_new_framework(
    monkeypatch, capsys, method, describe
):
    calls = []

    def selected(**options):
        calls.append(options)
        return dict(status="complete")

    monkeypatch.setattr(
        cli, "describe_minigrid_learnability_protocol", selected if describe else forbidden
    )
    monkeypatch.setattr(cli, "run_minigrid_learnability_pilot", forbidden if describe else selected)
    flags = [method if value == "bp" else value for value in CLI_FLAGS]
    monkeypatch.setattr(
        sys,
        "argv",
        ["pilot", *flags, "--joint-catalogue", "--train-a-rollout"]
        + (["--describe"] if describe else []),
    )
    cli.main()
    assert calls == [
        {**FLAGS, "qualification_method": method, "joint_catalogue": True, "train_a_rollout": True}
    ]
    assert json.loads(capsys.readouterr().out) == dict(status="complete")


@pytest.mark.parametrize("method", [None, "public_invariant_pc_relax64"])
@pytest.mark.parametrize("failed", [False, True])
def test_should_forward_only_selected_catalogue_and_preserve_paid_evaluator_receipt(
    monkeypatch, method, failed
):
    monkeypatch.setattr(rollout, "monotonic", lambda: 1.0)
    source, state = object(), object()
    arm = dict(source=source, states=[(0, object()), (128, state)])
    paid = live_report(method or "public_invariant_bp", status="failed" if failed else "complete")
    calls = []

    def evaluator(actual_source, actual_state, **options):
        assert actual_source is source and actual_state is state
        assert ("catalogue_method" in options) is (method is not None)
        if method is not None:
            assert options["catalogue_method"] == method
        assert options["updates"] == 128 and options["known_a_only"] is True
        assert options["budget"].budget.max_training_updates == 0
        assert options["budget"].budget.max_replay_examples == 0
        options["check_boundary"]()
        calls.append(options)
        if failed:
            raise evaluation.MiniGridEvaluationFailed(paid)
        return paid

    monkeypatch.setattr(rollout, "evaluate_neural_representation_checkpoint", evaluator)
    record, guard = rollout.run_train_rollout(
        arm,
        check_boundary=lambda: None,
        **({"catalogue_method": method} if method is not None else {}),
    )
    guard()
    assert len(calls) == 1
    assert record["status"] == ("failed" if failed else "complete")
    assert record["report"] == paid and record["report"] is not paid
    assert record["charged_updates"] == record["replay_examples"] == 0
    paid["work"]["live_prediction_calls_returned"] = 99
    assert record["report"]["work"]["live_prediction_calls_returned"] == 4
