"""Existing BP mixed objective through fixed taskspace and frozen fake serving.

One D12/H64 initializer draws1024 normal values, three A-only updates train6rows,
zero fits. Eight conditioning/residual/actor calls each score20rows;11 backward
calls cover26rows/12012 gradient values. Source/readouts35 forwards86rows plus
two fake prediction/control rows, one reset/close/name/limit read and3joint samples.
Other cases use cached owned reports/forks or stubs; no extra SGD or native physics.
"""

from copy import deepcopy
from dataclasses import asdict, replace
import json
from typing import Any, cast

import numpy as np
import pytest

from scripts import maniskill_continuous_smoke as smoke
from scripts import run_continuous_robot_learning as entry
from src.adapters import continuous_learnability_demo as demo
from src.adapters import numpy_continuous_bc as bp
from src.core import continuous_bc as core
from src.core import pick_cube_action_history as history
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)
from src.core.pick_cube_taskspace_history import PickCubeTaskSpaceHistoryRecipe
from src.infra import maniskill_continuous as native
from test_maniskill_joint_positions import StateEnvironment


def raw(index, offset=0.0):
    values = [0.0] * 42
    values[0], values[7], values[8], values[18] = index / 10, 0.02, 0.02, float(index > 0)
    values[36:42] = (0.05 + offset + index * 0.025, -0.025, 0.0125, -0.1, 0.025, 0.05)
    return tuple(values)


def episodes():
    spec = ContinuousActionSpec(42, native.ACTION_UNITS, (-1.0,) * 4, (1.0,) * 4)

    def episode(seed, actions):
        offset = 0.0 if seed == 17 else 0.025
        return ContinuousEpisode(
            f"pick-cube-readiness-teacher-{seed}",
            "pick-cube" if seed == 17 else "pick-cube-train-19",
            "train",
            spec,
            tuple(
                ContinuousTransition(
                    i,
                    raw(i, offset),
                    action,
                    float(i == len(actions) - 1),
                    raw(i + 1, offset),
                    i == len(actions) - 1,
                    False,
                )
                for i, action in enumerate(actions)
            ),
        )

    return (
        episode(17, ((0.5, -0.5, 0.25, -1.0), (-0.25, 0.5, -0.5, 1.0))),
        episode(19, ((0.25, 0.0, -0.5, 1.0), (-0.5, 0.25, 0.0, -1.0), (0.5, -0.25, 0.25, 1.0))),
    )


class TaskspaceEnvironment(StateEnvironment):
    def __init__(self):
        super().__init__()
        self.reset_observation = np.asarray([raw(0)], dtype=np.float64)

    def step(self, command):
        returned = super().step(command)
        return (np.asarray([raw(len(self.commands))], dtype=np.float64), *returned[1:])


def assert_state(first, second):
    assert first.policy == second.policy
    assert (
        first.completed_updates == second.completed_updates
        and first.training_examples == second.training_examples
    )
    for a, b in zip(
        first.parameters + first.velocity, second.parameters + second.velocity, strict=True
    ):
        np.testing.assert_array_equal(a, b)


def forbidden(*args, **kwargs):
    pytest.fail("invalid or stubbed binary request admitted model/native work")


@pytest.fixture(scope="module")
def learned(tmp_path_factory):
    warm, new = episodes()
    captured: dict[str, Any] = dict(inputs=[], outputs=[])
    train, runtime_factory, predict = (
        entry.run_continuous_learnability,
        demo.ActorShadowRuntime,
        bp.NumpyContinuousBCLearner.predict,
    )
    environment = TaskspaceEnvironment()

    def training(*args, **options):
        captured.update(training_inputs=args, training_options=options)
        result = train(*args, **options)
        captured["result"] = result
        assert result.candidate is not None
        captured["before_native"] = result.candidate.snapshot_state()
        return result

    def runtime(*args, **options):
        value = runtime_factory(*args, **options)
        captured.update(
            runtime=value,
            source=args[0],
            initial=args[0].snapshot_state(),
            budget=options["budget"],
            clock=options["clock"],
        )
        return value

    def predicting(owner, features):
        returned = predict(owner, features)
        if len(features) == 1:
            captured["inputs"].append(features.copy())
            captured["outputs"].append(returned.copy())
        return returned

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(entry, "run_continuous_learnability", training)
        patch.setattr(demo, "ActorShadowRuntime", runtime)
        patch.setattr(bp.NumpyContinuousBCLearner, "predict", predicting)
        patch.setattr(bp.observations, "fit_observation_standardization", forbidden)
        patch.setattr(native, "_make_environment", lambda: environment)
        patch.setattr(smoke, "version", lambda _: "fake-binary-gripper")
        report = entry.run_learning_rollout(
            tmp_path_factory.mktemp("binary-robot") / "case.json",
            warm,
            new,
            phase_updates=1,
            a_only_control=True,
            binary_gripper_axis=3,
            action_history_recipe=PickCubeTaskSpaceHistoryRecipe(),
        )
    assert report["complete"] is True, report
    return report, captured, environment, warm, new


def test_should_train_requested_existing_mixed_head_and_serve_the_same_frozen_candidate(learned):
    report, captured, _, warm, new = learned
    result = captured["result"]
    candidate = result.candidate
    assert (
        type(candidate) is bp.NumpyContinuousBCLearner and candidate.policy.binary_gripper_axis == 3
    )
    assert candidate.policy.hidden_dim == 64 and candidate.policy.seed == 17
    assert (
        candidate.policy.spec.observation_dim == 12
        and candidate.policy.observation_standardization is None
    )
    assert (
        report["binary_gripper_axis"] == 3 and report["training_objective"] == core.MIXED_OBJECTIVE
    )
    assert report["protocol_id"].endswith("_binary_gripper_train_v1")
    assert report["candidate_unchanged"] is True
    assert_state(captured["before_native"], candidate.snapshot_state())
    assert_state(captured["initial"], captured["source"].snapshot_state())
    assert_state(captured["initial"], captured["runtime"].actor.snapshot_state().state)
    assert candidate.snapshot_state().completed_updates == captured["budget"].updates_completed == 3
    assert (
        candidate.snapshot_state().training_examples == 6
        and captured["budget"].replay_examples_completed == 0
    )
    assert captured["clock"].now() == 5 and result.report["progress"]["next_tick"] == 6
    assert len(new.transitions) == 3 and len(warm.transitions) == 2
    assert captured["training_options"] == dict(
        learner_family="bp",
        phase_updates=1,
        a_only_control=True,
        binary_gripper_axis=3,
        action_history_recipe=PickCubeTaskSpaceHistoryRecipe(),
    )


def test_should_keep_a_only_mixed_receipts_and_read_only_b_diagnostics_with_truthful_costs(learned):
    report, captured, _, warm, new = learned
    training = report["training"]
    assert training["settings"]["training_contexts"] == ["warm"] * 3
    assert training["work"]["training_rows_by_context"] == dict(warm=6, new=0)
    assert training["work"]["diagnostic_rows_by_context"] == dict(warm=8, new=12)
    assert (
        training["work"]["optimizer_updates"] == 3
        and training["work"]["binary_gripper_loss_rows"] == 26
    )
    assert (
        training["work"]["fit_calls"]
        == training["work"]["fit_rows"]
        == training["work"]["statistics_numeric_bytes_per_owner"]
        == 0
    )
    assert [row["episode_id"] for row in training["applied_updates"]] == [warm.episode_id] * 3
    assert all(
        row["diagnostic"]["definition"] == core.MIXED_OBJECTIVE
        for row in training["applied_updates"]
    )
    for checkpoint in training["checkpoints"]:
        assert checkpoint["contexts"]["new"]["episode_id"] == new.episode_id
        for context in checkpoint["contexts"].values():
            assert (
                context["conditioning"]["conditioning"]["objective_definition"]
                == core.MIXED_OBJECTIVE
            )
    assert captured["result"].report == training
    json.dumps(report, allow_nan=False)


def test_should_predict_causal_twelve_features_but_preserve_raw42_and_soft_commands(learned):
    report, captured, environment, _, _ = learned
    native_report = report["rollout"]
    assert (
        native_report["action_history_exact"]
        is native_report["frozen_state_exact"]
        is native_report["original_owner_exact"]
        is True
    )
    assert native_report["joint_position_exact"] is True
    assert native_report["episode"]["spec"]["observation_dim"] == 42
    assert native_report["learner_policy"]["binary_gripper_axis"] == 3
    values = captured["inputs"]
    assert len(values) == 2 and values[0].shape == values[1].shape == (1, 12)
    assert tuple(values[0][0]) == history.action_history_observation(
        raw(0), history.ActionHistoryState(), PickCubeTaskSpaceHistoryRecipe()
    )
    first_action = tuple(native_report["episode"]["transitions"][0]["action"])
    state = history.advance_action_history(history.ActionHistoryState(), first_action)
    assert tuple(values[1][0]) == history.action_history_observation(
        raw(1), state, PickCubeTaskSpaceHistoryRecipe()
    )
    expected = np.concatenate(captured["outputs"]).astype(np.float32)
    np.testing.assert_array_equal(np.concatenate(environment.commands), expected)
    assert np.any(np.abs(expected[:, 3]) < 1.0)
    assert len(environment.commands) == 2 and environment.close_calls == 1
    assert environment.events.count("names") == environment.events.count("limits") == 1
    assert native_report["joint_position_work"]["samples"] == 3


@pytest.mark.parametrize(
    "family,axis",
    [
        ("pc", 3),
        ("neutral-cpc", 3),
        ("active-cpc", 3),
        ("bp", True),
        ("bp", 3.0),
        ("bp", 1),
        ("bp", -1),
    ],
)
def test_should_refuse_non_bp_or_ambiguous_axis_before_input_admission_or_training(
    monkeypatch, tmp_path, family, axis
):
    monkeypatch.setattr(entry, "_admit_inputs", forbidden)
    monkeypatch.setattr(entry, "run_continuous_learnability", forbidden)
    with pytest.raises(ValueError):
        entry.run_learning_rollout(
            tmp_path / "invalid.json",
            cast(Any, object()),
            cast(Any, object()),
            learner_family=family,
            binary_gripper_axis=axis,
        )


@pytest.mark.parametrize("which", [0, 1])
def test_should_refuse_nonbinary_target_in_either_train_episode_even_when_b_is_read_only(
    monkeypatch, tmp_path, which
):
    records = list(episodes())
    episode = records[which]
    row = replace(episode.transitions[0], action=(*episode.transitions[0].action[:3], 0.5))
    records[which] = replace(episode, transitions=(row, *episode.transitions[1:]))
    monkeypatch.setattr(demo, "NumpyContinuousBCLearner", forbidden)
    monkeypatch.setattr(demo, "fit_training_standardization", forbidden)
    monkeypatch.setattr(entry, "run_frozen_rollout", forbidden)
    report = entry.run_learning_rollout(
        tmp_path / "invalid-target.json",
        *records,
        phase_updates=1,
        a_only_control=True,
        binary_gripper_axis=3,
        action_history_recipe=PickCubeTaskSpaceHistoryRecipe(),
    )
    assert (
        report["complete"] is False
        and report["rollout_attempted"] is False
        and report["training"] is None
    )
    assert "binary" in report["error"].lower()
    assert (tmp_path / "invalid-target.json").is_file()


def cloned_result(learned):
    original = learned[1]["result"]
    return demo.ContinuousLearnabilityResult(deepcopy(original.report), original.candidate.fork())


@pytest.mark.parametrize("fault", ["missing_axis", "unrequested_axis", "reported_axis"])
def test_should_bind_requested_and_unrequested_policy_before_serving_and_keep_paid_report(
    learned, monkeypatch, tmp_path, fault
):
    result = cloned_result(learned)
    if fault == "missing_axis":
        candidate = result.candidate
        state = candidate.snapshot_state()
        policy = replace(state.policy, binary_gripper_axis=None)
        candidate._policy = policy
        candidate.restore_state(replace(state, policy=policy))
        result.report["policy"] = asdict(policy)
    elif fault == "reported_axis":
        result.report["policy"]["binary_gripper_axis"] = None
    monkeypatch.setattr(entry, "run_continuous_learnability", lambda *args, **options: result)
    monkeypatch.setattr(entry, "run_frozen_rollout", forbidden)
    options: dict[str, Any] = {} if fault == "unrequested_axis" else dict(binary_gripper_axis=3)
    report = entry.run_learning_rollout(
        tmp_path / "policy.json",
        learned[3],
        learned[4],
        phase_updates=1,
        a_only_control=True,
        action_history_recipe=PickCubeTaskSpaceHistoryRecipe(),
        **options,
    )
    assert report["complete"] is False and report["rollout_attempted"] is False
    assert (
        report["training"] == result.report
        and report["training"]["progress"]["updates_completed"] == 3
    )
    if fault == "unrequested_axis":
        # This gate must bind the policy even before any A-only history checks.
        with pytest.raises(ValueError, match="gripper likelihood"):
            entry._admit_candidate(
                result, learned[1]["training_inputs"][0], learned[1]["training_inputs"][1], "bp", 1
            )


@pytest.mark.parametrize("objective", [core.OBJECTIVE, None])
def test_should_refuse_coherently_changed_or_missing_a_only_receipt_objective_before_native(
    learned, monkeypatch, tmp_path, objective
):
    result = cloned_result(learned)
    for row in (result.report["applied_updates"][0], result.report["events"][0]["applied"][0]):
        if objective is None:
            row["diagnostic"].pop("definition")
        else:
            row["diagnostic"]["definition"] = objective
    monkeypatch.setattr(entry, "run_continuous_learnability", lambda *args, **options: result)
    monkeypatch.setattr(entry, "run_frozen_rollout", forbidden)
    report = entry.run_learning_rollout(
        tmp_path / "receipt.json",
        learned[3],
        learned[4],
        phase_updates=1,
        a_only_control=True,
        binary_gripper_axis=3,
        action_history_recipe=PickCubeTaskSpaceHistoryRecipe(),
    )
    assert report["complete"] is False and report["rollout_attempted"] is False
    assert report["training"] == result.report and len(report["training"]["applied_updates"]) == 3


@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("visual", [False, True])
def test_should_forward_cli_axis_only_when_selected_and_keep_readiness_dispatch(
    monkeypatch, tmp_path, selected, visual
):
    reads, calls = [], []

    def reader(path, seed, **options):
        reads.append((seed, options))
        return episodes()[0 if seed == 17 else 1], {}

    monkeypatch.setattr(entry, "_read_readiness_teacher", reader)
    monkeypatch.setattr(entry, "_read_teacher", forbidden)

    def run_stub(*args, **options):
        calls.append(options)
        return dict(complete=False)

    monkeypatch.setattr(entry, "run_learning_rollout", run_stub)
    flags = (["--binary-gripper"] if selected else []) + (
        ["--visual-observation"] if visual else []
    )
    assert (
        entry.main(
            [
                "--output",
                str(tmp_path / "cli.json"),
                "--warm-teacher",
                "A.json",
                "--new-teacher",
                "B.json",
                "--readiness-teachers",
                *flags,
            ]
        )
        == 1
    )
    assert reads == [(17, dict(visual_observation=visual)), (19, dict(visual_observation=visual))]
    assert (
        (calls[0].get("binary_gripper_axis") == 3)
        if selected
        else ("binary_gripper_axis" not in calls[0])
    )


@pytest.mark.parametrize("family", ["pc", "neutral-cpc", "active-cpc"])
def test_should_refuse_cli_non_bp_before_any_file_read_or_runner(monkeypatch, tmp_path, family):
    monkeypatch.setattr(entry, "_read_teacher", forbidden)
    monkeypatch.setattr(entry, "_read_readiness_teacher", forbidden)
    monkeypatch.setattr(entry, "run_learning_rollout", forbidden)
    with pytest.raises(SystemExit) as error:
        entry.main(
            [
                "--output",
                str(tmp_path / "invalid.json"),
                "--warm-teacher",
                "A.json",
                "--new-teacher",
                "B.json",
                "--learner-family",
                family,
                "--binary-gripper",
            ]
        )
    assert error.value.code == 2
