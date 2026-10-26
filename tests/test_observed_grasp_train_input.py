"""Explicit observed warm-teacher admission and one ordinary A-only BP run.

One shared D12/H64/A4 initializer draws1024 normal values;3 mixed updates/6A
rows, zero fits. Eight residual/conditioning/actor calls per group score16rows.
11 backwards/22rows/12012 gradient values;35 learner forwards/70rows plus two
fake predictions=37/72.29 normalizations/56rows/672coordinates;101888 total
derived dense MACs. One fake scene/two controls/three joint samples; no physics.
All other runner calls are stubbed and all metadata cases use literal records.
"""

from copy import deepcopy
from dataclasses import asdict, fields, replace
import json
from typing import Any

import numpy as np
import pytest

from scripts import maniskill_continuous_smoke as smoke
from scripts import run_continuous_robot_learning as robot
from scripts import run_native_stream_demo as reader
from src.adapters import continuous_learnability_demo as demo
from src.adapters import numpy_continuous_bc as bp
from src.core import continuous_bc as core
from src.core import pick_cube_action_history as history
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)
from src.core.pick_cube_readiness_teacher import DEFINITION, OBSERVED_GRASP_DEFINITION
from src.core.pick_cube_taskspace_history import PickCubeTaskSpaceHistoryRecipe
from src.core.readiness_stage_diagnostics import (
    ReadinessStageBinding,
    copy_readiness_stage_bindings,
    readiness_stage_metadata,
)
from src.infra import maniskill_continuous as native
from src.infra.maniskill_task_teacher import ObservedGraspTaskTeacherResult, TaskTeacherResult
from test_maniskill_joint_positions import StateEnvironment


RECIPE = PickCubeTaskSpaceHistoryRecipe()


def raw(index, seed=17):
    values = [0.0] * 42
    values[0], values[7], values[8] = index / 100, 0.02, 0.02
    values[18] = float(seed == 17 and index == 0)
    values[22], values[32] = 1.0, 1.0
    values[36:42] = (
        0.05 + 0.025 * index + (0.05 if seed == 19 else 0.0),
        -0.025,
        0.0125,
        -0.1,
        0.025,
        0.05,
    )
    return tuple(values)


def receipt(seed=17, *, observed=True):
    stages = ("lift", "align") if observed else ("descend", "grasp")
    actions = (
        ((0.5, 0.0, 0.5, -1.0), (0.25, 0.5, 0.0, 1.0))
        if observed
        else (
            (0.0, 0.0, -0.5, 1.0),
            (0.0, 0.0, 0.0, -1.0),
        )
    )
    episode = ContinuousEpisode(
        f"pick-cube-{'observed-grasp' if observed else 'readiness'}-teacher-{seed}",
        "pick-cube" if seed == 17 else "pick-cube-train-19",
        "train",
        ContinuousActionSpec(42, native.ACTION_UNITS, (-1.0,) * 4, (1.0,) * 4),
        tuple(
            ContinuousTransition(
                i, raw(i, seed), action, float(i == 1), raw(i + 1, seed), i == 1, False
            )
            for i, action in enumerate(actions)
        ),
    )
    result_type = ObservedGraspTaskTeacherResult if observed else TaskTeacherResult
    options: dict[str, Any] = dict(observed_grasp=(True, False)) if observed else {}
    # Selected19 uses a valid observed transport/approach trajectory too.
    if observed and seed == 19:
        rows = []
        for row in episode.transitions:
            before, after = list(row.observation), list(row.next_observation)
            before[18] = float(row.step == 0)
            after[18] = 0.0
            rows.append(replace(row, observation=tuple(before), next_observation=tuple(after)))
        episode = replace(episode, transitions=tuple(rows))
        options["observed_grasp"] = (True, False)
    result = result_type(
        episode,
        True,
        True,
        1,
        2,
        2,
        2,
        stages,
        native.ManiSkillWork(2, 2, 10, 0),
        None,
        None,
        teacher_definition=OBSERVED_GRASP_DEFINITION if observed else DEFINITION,
        **options,
    )
    return json.loads(
        json.dumps(
            dict(
                asdict(result),
                qualified=True,
                seed=seed,
                context=episode.context,
                role="train",
                task=native.TASK_ID,
                controller=native.CONTROL_MODE,
                optimizer_updates=0,
            )
        )
    )


def write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf8")
    return path


def forbidden(*args, **kwargs):
    pytest.fail("metadata/stub case admitted fitting, model or native work")


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    directory = tmp_path_factory.mktemp("observed-train-input")
    result = {}
    for name, seed, selected in (("warm", 17, True), ("selected19", 19, True), ("new", 19, False)):
        payload = receipt(seed, observed=selected)
        path = write(directory / f"{name}.json", payload)
        episode, metadata = reader._read_readiness_teacher(
            path, seed, **({"observed_grasp": True} if selected else {})
        )
        result[name] = (payload, path, episode, metadata)
    return result


@pytest.mark.parametrize("name", ["warm", "selected19"])
def test_should_admit_owned_selected_train_receipts_and_original_acquisition_metadata(inputs, name):
    payload, _, episode, metadata = inputs[name]
    assert episode.role == "train" and episode.spec.observation_dim == 42
    assert episode.episode_id == payload["candidate_episode"]["episode_id"]
    assert metadata["mode"] == "observed_grasp_readiness"
    assert metadata["teacher_definition"] == OBSERVED_GRASP_DEFINITION
    assert metadata["observed_grasp"] == [True, False]
    assert metadata["observed_grasp"] is not payload["observed_grasp"]
    assert metadata["stages"] == ["lift", "align"]
    assert metadata["teacher_work"] == payload["work"]
    assert (
        metadata["teacher_geometry_attempts"]
        == metadata["teacher_geometry_returns"]
        == metadata["teacher_actions"]
        == 2
    )
    assert metadata["per_file_byte_limit"] == 1024**2
    assert metadata["source_file_bytes"] == len(json.dumps(payload).encode("utf8"))
    assert metadata["acquisition_scope"] == "retained_teacher_work_not_new_simulator_work"


def test_should_keep_unselected_v1_metadata_schema_and_definition_unchanged(inputs):
    _, _, episode, metadata = inputs["new"]
    assert episode.episode_id == "pick-cube-readiness-teacher-19"
    assert metadata["mode"] == "geometry_readiness" and metadata["teacher_definition"] == DEFINITION
    assert "observed_grasp" not in metadata
    assert tuple(row.action[3] for row in episode.transitions) == (1.0, -1.0)


@pytest.mark.parametrize(
    "damage",
    [
        "missing_feedback",
        "integer_feedback",
        "float_feedback",
        "short_feedback",
        "mismatched_feedback",
        "current_bool",
        "current_integer",
        "next_fraction",
        "nonfinite",
        "dimension_float",
        "old_definition",
        "old_id",
        "final_role",
        "probe_role",
        "guard",
        "seed_bool",
        "work_bool",
        "geometry_float",
        "false_qualified",
        "wrong_stage",
    ],
)
def test_should_refuse_malformed_selected_evidence_before_any_learner(tmp_path, damage):
    payload = receipt()
    episode = payload["candidate_episode"]
    if damage == "missing_feedback":
        del payload["observed_grasp"]
    elif damage in ("integer_feedback", "float_feedback", "short_feedback", "mismatched_feedback"):
        payload["observed_grasp"] = {
            "integer_feedback": [1, False],
            "float_feedback": [1.0, False],
            "short_feedback": [True],
            "mismatched_feedback": [False, False],
        }[damage]
    elif damage in ("current_bool", "current_integer", "nonfinite"):
        episode["transitions"][0]["observation"][18] = {
            "current_bool": True,
            "current_integer": 1,
            "nonfinite": float("nan"),
        }[damage]
    elif damage == "next_fraction":
        episode["transitions"][-1]["next_observation"][18] = 0.5
    elif damage == "dimension_float":
        episode["spec"]["observation_dim"] = 42.0
    elif damage == "old_definition":
        payload["teacher_definition"] = DEFINITION
    elif damage == "old_id":
        episode["episode_id"] = "pick-cube-readiness-teacher-17"
    elif damage in ("final_role", "probe_role"):
        payload["role"] = episode["role"] = (
            "final_test" if damage == "final_role" else "probe_train"
        )
    elif damage == "guard":
        payload["guard_seed"] = 31
    elif damage == "seed_bool":
        payload["seed"] = True
    elif damage == "work_bool":
        payload["work"]["possible_failed_physics_steps"] = False
    elif damage == "geometry_float":
        payload["geometry_returns"] = 2.0
    elif damage == "false_qualified":
        payload["qualified"] = 1
    else:
        payload["stages"][0] = "grasp"
        episode["transitions"][0]["action"] = [0.0, 0.0, 0.0, -1.0]
    with pytest.raises(ValueError):
        reader._read_readiness_teacher(
            write(tmp_path / "bad.json", payload), 17, observed_grasp=True
        )


@pytest.mark.parametrize("marker", [None, [True, False]])
def test_should_refuse_observed_marker_presence_even_when_definition_and_id_are_forged_v1(
    tmp_path, marker
):
    payload = receipt(observed=False)
    payload["observed_grasp"] = marker
    with pytest.raises(ValueError):
        reader._read_readiness_teacher(write(tmp_path / "forged.json", payload), 17)


@pytest.mark.parametrize("fault", ["integer", "none", "alias", "visual"])
def test_should_refuse_invalid_reader_mode_before_opening_a_file(monkeypatch, tmp_path, fault):
    monkeypatch.setattr(reader, "_teacher_payload", forbidden)
    options: dict[str, Any] = dict(observed_grasp=True)
    if fault != "visual":
        options["observed_grasp"] = {"integer": 1, "none": None, "alias": np.bool_(True)}[fault]
    else:
        options["visual_observation"] = True
    with pytest.raises(ValueError):
        reader._read_readiness_teacher(tmp_path / "missing.json", 17, **options)


def bindings(inputs):
    return tuple(
        ReadinessStageBinding(inputs[name][2], tuple(inputs[name][3]["stages"]))
        for name in ("warm", "new")
    )


@pytest.mark.parametrize("mode", ["raw42", "history47", "taskspace12"])
def test_should_keep_two_field_stage_bindings_and_per_record_definitions_through_views(
    inputs, mode
):
    pair = bindings(inputs)
    if mode != "raw42":
        recipe = history.PickCubeActionHistoryRecipe() if mode == "history47" else RECIPE
        pair = tuple(
            ReadinessStageBinding(
                history.action_history_training_episode(value.episode, recipe), value.stages
            )
            for value in pair
        )
    expected = tuple(value.episode for value in pair)
    owned = copy_readiness_stage_bindings(pair, expected)
    assert [field.name for field in fields(ReadinessStageBinding)] == ["episode", "stages"]
    assert all(
        value is not original and value.episode is not original.episode
        for value, original in zip(owned, pair, strict=True)
    )
    metadata = readiness_stage_metadata(owned)
    assert metadata["warm"]["definition"] == OBSERVED_GRASP_DEFINITION
    assert metadata["new"]["definition"] == DEFINITION
    assert metadata["warm"]["stages"] == ("lift", "align")
    assert metadata["new"]["stages"] == ("descend", "grasp")
    assert metadata["warm"]["rows"] == metadata["new"]["rows"] == 2
    metadata["warm"]["stages"] = ("settle", "settle")
    assert readiness_stage_metadata(owned)["warm"]["stages"] == ("lift", "align")


@pytest.mark.parametrize("damage", ["reverse", "changed_row", "role", "flag_alias"])
def test_should_revalidate_original_stage_pair_order_roles_and_full_row_equality(inputs, damage):
    pair = deepcopy(bindings(inputs))
    episodes = tuple(value.episode for value in pair)
    if damage == "reverse":
        pair, episodes = tuple(reversed(pair)), tuple(reversed(episodes))
    elif damage == "changed_row":
        warm = episodes[0]
        rows = (replace(warm.transitions[0], reward=0.5), warm.transitions[1])
        episodes = (replace(warm, transitions=rows), episodes[1])
    elif damage == "role":
        episodes = (replace(episodes[0], role="probe_train"), episodes[1])
    else:
        row = pair[0].episode.transitions[0]
        values = list(row.observation)
        values[18] = True
        object.__setattr__(row, "observation", tuple(values))
    with pytest.raises(ValueError):
        copy_readiness_stage_bindings(pair, episodes)


@pytest.mark.parametrize("selected", [False, True])
def test_should_forward_selected_true_only_to_warm_reader_and_preserve_default_callshape(
    inputs, monkeypatch, tmp_path, selected
):
    calls, runs = [], []

    def read(path, seed, **options):
        calls.append((seed, options))
        if seed == 17 and not selected:
            payload = receipt(observed=False)
            episode = reader._teacher_episode(
                payload["candidate_episode"],
                "pick-cube",
                "pick-cube-readiness-teacher-17",
                visual_observation=False,
            )
            return episode, dict(stages=payload["stages"])
        value = inputs["warm" if seed == 17 else "new"]
        return value[2], value[3]

    def run(*args, **options):
        runs.append((args, options))
        return dict(complete=True)

    monkeypatch.setattr(robot, "_read_readiness_teacher", read)
    monkeypatch.setattr(robot, "run_learning_rollout", run)
    args = [
        "--output",
        str(tmp_path / "stub.json"),
        "--warm-teacher",
        "warm.json",
        "--new-teacher",
        "new.json",
        "--readiness-teachers",
        "--readiness-stage-residuals",
        "--phase-updates",
        "1",
    ]
    if selected:
        args += ["--observed-grasp-warm-teacher", "--a-only-control"]
    assert robot.main(args) == 0
    assert calls == [
        (17, dict(visual_observation=False, **({"observed_grasp": True} if selected else {}))),
        (19, dict(visual_observation=False)),
    ]
    options = runs[0][1]
    assert (options.get("a_only_control") is True) is selected
    assert "observed_grasp" not in options and "observed_grasp_warm_teacher" not in options
    assert readiness_stage_metadata(options["diagnostic_stage_bindings"])["warm"]["definition"] == (
        OBSERVED_GRASP_DEFINITION if selected else DEFINITION
    )


@pytest.mark.parametrize(
    "extra",
    [
        [],
        ["--readiness-teachers"],
        ["--a-only-control"],
        ["--readiness-teachers", "--a-only-control", "--visual-observation"],
        ["--readiness-teachers", "--a-only-control", "--lighting-drift"],
    ],
)
def test_should_refuse_invalid_cli_scope_before_any_teacher_read_or_output(
    monkeypatch, tmp_path, extra
):
    monkeypatch.setattr(robot, "_read_readiness_teacher", forbidden)
    monkeypatch.setattr(robot, "_read_teacher", forbidden)
    monkeypatch.setattr(robot, "run_learning_rollout", forbidden)
    output = tmp_path / "not-created.json"
    with pytest.raises(SystemExit) as raised:
        robot.main(
            [
                "--output",
                str(output),
                "--warm-teacher",
                "missing-A",
                "--new-teacher",
                "missing-B",
                "--observed-grasp-warm-teacher",
                *extra,
            ]
        )
    assert raised.value.code == 2 and not output.exists()


class TaskspaceEnvironment(StateEnvironment):
    def __init__(self):
        super().__init__()
        self.reset_observation = np.asarray([raw(0)], dtype=np.float64)

    def step(self, command):
        returned = super().step(command)
        return (np.asarray([raw(len(self.commands))], dtype=np.float64), *returned[1:])


def assert_state(first, second):
    assert first.policy == second.policy
    assert (first.completed_updates, first.training_examples) == (
        second.completed_updates,
        second.training_examples,
    )
    for a, b in zip(
        first.parameters + first.velocity, second.parameters + second.velocity, strict=True
    ):
        np.testing.assert_array_equal(a, b)


@pytest.fixture(scope="module")
def learned(inputs, tmp_path_factory):
    captured: dict[str, Any] = dict(predictions=[], reads=[])
    train, runtime_factory, predict = (
        robot.run_continuous_learnability,
        demo.ActorShadowRuntime,
        bp.NumpyContinuousBCLearner.predict,
    )
    read = robot._read_readiness_teacher
    environment = TaskspaceEnvironment()
    output = tmp_path_factory.mktemp("observed-train-main") / "result.json"

    def reading(path, seed, **options):
        captured["reads"].append((seed, options))
        return read(path, seed, **options)

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
            captured["predictions"].append((features.copy(), returned.copy()))
        return returned

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(robot, "_read_readiness_teacher", reading)
        patch.setattr(robot, "run_continuous_learnability", training)
        patch.setattr(demo, "ActorShadowRuntime", runtime)
        patch.setattr(bp.NumpyContinuousBCLearner, "predict", predicting)
        patch.setattr(bp.observations, "fit_observation_standardization", forbidden)
        patch.setattr(native, "_make_environment", lambda: environment)
        patch.setattr(smoke, "version", lambda _: "fake-observed-train")
        status = robot.main(
            [
                "--output",
                str(output),
                "--warm-teacher",
                str(inputs["warm"][1]),
                "--new-teacher",
                str(inputs["new"][1]),
                "--readiness-teachers",
                "--observed-grasp-warm-teacher",
                "--readiness-stage-residuals",
                "--a-only-control",
                "--phase-updates",
                "1",
                "--binary-gripper",
                "--taskspace-history",
                "executed",
            ]
        )
    report = json.loads(output.read_text(encoding="utf8"))
    assert status == 0 and report["complete"] is True, report
    return report, captured, environment


def test_should_train_only_selected_a_then_serve_unchanged_full_candidate_with_read_only_b(learned):
    report, captured, _ = learned
    candidate = captured["result"].candidate
    assert type(candidate) is bp.NumpyContinuousBCLearner
    assert candidate.policy.hidden_dim == 64 and candidate.policy.spec.observation_dim == 12
    assert (
        candidate.policy.binary_gripper_axis == 3
        and candidate.policy.observation_standardization is None
    )
    assert (
        candidate.snapshot_state().completed_updates,
        candidate.snapshot_state().training_examples,
    ) == (3, 6)
    assert_state(captured["before_native"], candidate.snapshot_state())
    assert_state(captured["initial"], captured["source"].snapshot_state())
    assert_state(captured["initial"], captured["runtime"].actor.snapshot_state().state)
    assert (
        captured["budget"].updates_completed == 3
        and captured["budget"].replay_examples_completed == 0
    )
    assert captured["reads"] == [
        (17, dict(visual_observation=False, observed_grasp=True)),
        (19, dict(visual_observation=False)),
    ]
    assert report["candidate_unchanged"] is True and report["a_only_control"] is True
    training = report["training"]
    assert training["work"]["training_rows_by_context"] == dict(warm=6, new=0)
    assert training["work"]["diagnostic_rows_by_context"] == dict(warm=8, new=8)
    assert [row["episode_id"] for row in training["applied_updates"]] == [
        "pick-cube-observed-grasp-teacher-17"
    ] * 3
    assert all(
        row["diagnostic"]["definition"] == core.MIXED_OBJECTIVE
        for row in training["applied_updates"]
    )
    assert training["settings"]["probe_reads"] == training["settings"]["final_reads"] == 0


def test_should_bind_all_eight_physical_residual_stage_and_axis_partitions_to_original_versions(
    learned,
):
    report, _, _ = learned
    training = report["training"]
    assert training["diagnostic_stage_sources"]["warm"]["definition"] == OBSERVED_GRASP_DEFINITION
    assert training["diagnostic_stage_sources"]["new"]["definition"] == DEFINITION
    assert [row["updates"] for row in training["checkpoints"]] == [0, 1, 2, 3]
    for checkpoint in training["checkpoints"]:
        assert checkpoint["state_exact"] is checkpoint["owners_exact"] is True
        for name, labels in (("warm", ("lift", "align")), ("new", ("descend", "grasp"))):
            residual = checkpoint["contexts"][name]["residual"]
            assert residual["prediction_calls"] == 1 and residual["prediction_rows"] == 2
            groups = residual["residuals"]["stages"]
            assert [
                (row["label"], row["row_indices"], row["summary"]["rows"]) for row in groups
            ] == [(labels[0], [0], 1), (labels[1], [1], 1)]
            for row in groups:
                assert [axis["index"] for axis in row["summary"]["axes"]] == [0, 1, 2, 3]
                assert [axis["unit"] for axis in row["summary"]["axes"]] == list(
                    native.ACTION_UNITS
                )
    work = training["work"]
    assert work["stage_residual_summaries"] == 16
    assert work["optimizer_updates"] == 3 and work["training_rows"] == 6
    assert work["binary_gripper_loss_rows"] == 22
    assert work["fit_calls"] == work["fit_rows"] == work["fit_feature_values"] == 0
    assert work["forward_passes"] == 35 and work["forward_rows"] == 70
    assert work["backward_passes"] == 11 and work["gradient_values"] == 12012
    assert work["dense_multiply_accumulates"] == 99840


def test_should_use_actual_causal_taskspace_inputs_but_keep_raw42_and_joint_telemetry_in_frozen_return(
    learned,
):
    report, captured, environment = learned
    native_report = report["rollout"]
    assert (
        native_report["joint_position_exact"]
        is native_report["action_history_exact"]
        is native_report["frozen_state_exact"]
        is native_report["original_owner_exact"]
        is True
    )
    rows = native_report["episode"]["transitions"]
    assert native_report["episode"]["spec"]["observation_dim"] == 42 and len(rows) == 2
    predictions = captured["predictions"]
    assert len(predictions) == 2
    state = history.ActionHistoryState()
    for index, ((features, outputs), row, command) in enumerate(
        zip(predictions, rows, environment.commands, strict=True)
    ):
        assert tuple(features[0]) == history.action_history_observation(raw(index), state, RECIPE)
        np.testing.assert_array_equal(command, outputs.astype(np.float32))
        assert tuple(row["observation"]) == raw(index)
        state = history.advance_action_history(state, tuple(row["action"]))
    assert native_report["source_updates"] == 3 and native_report["source_training_rows"] == 6
    assert "corrections" not in native_report
    assert environment.close_calls == 1 and len(environment.commands) == 2
    assert environment.events.count("names") == environment.events.count("limits") == 1
    assert native_report["joint_position_work"]["samples"] == 3
