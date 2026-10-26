"""Qualified readiness JSON admission and existing CLI composition.

One shared state42 BP phase-one run: one initializer/2944 normal values,
three updates/six rows, one A-only fit/two rows/84 values. Eight conditioning,
residual and actor-readout calls each score sixteen rows; frozen fake IO adds
two prediction/control rows, one reset/close/name/limit read and three joint
samples. The module makes 84 capped payload reads (69 state/15 visual, including
two literal handles) and nine small verification reads. Public21 CLI routing
validates 6312 source coordinates and returns 168 projected values. All other
runner calls are stubbed. No native physics or collection.
"""

from dataclasses import asdict
import io
import json
from pathlib import Path
import sys
from typing import Any

import pytest

from scripts import maniskill_continuous_smoke as smoke
from scripts import run_continuous_robot_learning as robot
from scripts import run_native_stream_demo as cli
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)
from src.core.pick_cube_readiness_teacher import DEFINITION
from src.core.pick_cube_rgb_features import pick_cube_rgb_recipe
from src.infra import maniskill_continuous as native
from src.infra.maniskill_task_teacher import TaskTeacherResult
from test_maniskill_joint_positions import StateEnvironment


MISSING = object()


def receipt(seed=17, *, visual=False):
    dimension = 789 if visual else 42
    offset = 0.0 if seed == 17 else 10.0

    def observation(index):
        return (
            (offset + index, float(seed == 19))
            + (0.0,) * ((21 if visual else 42) - 2)
            + ((0.25,) * 768 if visual else ())
        )

    context = "pick-cube" if seed == 17 else "pick-cube-train-19"
    spec = ContinuousActionSpec(dimension, native.ACTION_UNITS, (-1.0,) * 4, (1.0,) * 4)
    actions = ((0.0, 0.0, -0.8, 1.0), (0.0, 0.0, 0.0, -1.0))
    value = ContinuousEpisode(
        f"pick-cube-readiness-teacher-{seed}",
        context,
        "train",
        spec,
        tuple(
            ContinuousTransition(
                i, observation(i), actions[i], float(i == 1), observation(i + 1), i == 1, False
            )
            for i in range(2)
        ),
    )
    visual_work = (
        native.ManiSkillVisualWork(3, 3, 147456, 2367, 147456, 2367, 18936, 49152, 0, 0)
        if visual
        else None
    )
    result = TaskTeacherResult(
        value,
        True,
        True,
        1,
        2,
        2,
        2,
        ("descend", "grasp"),
        native.ManiSkillWork(2, 2, 10, 0),
        None,
        None,
        observation_recipe=pick_cube_rgb_recipe() if visual else None,
        visual_work=visual_work,
        teacher_definition=DEFINITION,
    )
    return json.loads(
        json.dumps(
            dict(
                asdict(result),
                qualified=True,
                seed=seed,
                context=context,
                role="train",
                task=native.TASK_ID,
                controller=native.CONTROL_MODE,
            )
        )
    )


def write(path, raw):
    path.write_text(json.dumps(raw), encoding="utf8")
    return path


def change(raw, path, value):
    owner = raw
    for key in path[:-1]:
        owner = owner[key]
    if value is MISSING:
        del owner[path[-1]]
    else:
        owner[path[-1]] = value


def forbidden(*args, **kwargs):
    pytest.fail("metadata admission or stub routing executed an unexpected runner")


@pytest.mark.parametrize("seed", [17, 19])
@pytest.mark.parametrize("visual", [False, True])
def test_should_release_owned_readiness_pairs_and_retained_acquisition_costs(
    tmp_path, seed, visual
):
    raw = receipt(seed, visual=visual)
    path = write(tmp_path / "teacher.json", raw)
    original = path.read_bytes()
    episode, metadata = cli._read_readiness_teacher(path, seed, visual_observation=visual)
    assert episode.role == "train" and episode.context == raw["context"]
    assert episode.episode_id == f"pick-cube-readiness-teacher-{seed}"
    assert episode.spec.observation_dim == (789 if visual else 42)
    assert tuple(row.action for row in episode.transitions) == (
        (0.0, 0.0, -0.8, 1.0),
        (0.0, 0.0, 0.0, -1.0),
    )
    expected = dict(
        mode="geometry_readiness",
        teacher_definition=DEFINITION,
        seed=seed,
        context=raw["context"],
        episode_id=episode.episode_id,
        stages=raw["stages"],
        teacher_work=raw["work"],
        teacher_visual_work=raw["visual_work"],
        teacher_creation_attempts=1,
        teacher_geometry_attempts=2,
        teacher_geometry_returns=2,
        teacher_actions=2,
        source_file_bytes=len(original),
        per_file_byte_limit=(4 if visual else 1) * 1024**2,
        acquisition_scope="retained_teacher_work_not_new_simulator_work",
    )
    assert json.loads(json.dumps(metadata)) == expected
    metadata["teacher_work"]["completed_steps"] = 99
    metadata["stages"][0] = "align"
    object.__setattr__(episode.transitions[0], "reward", 0.5)
    again, retained = cli._read_readiness_teacher(path, seed, visual_observation=visual)
    assert retained["teacher_work"]["completed_steps"] == 2
    assert retained["stages"][0] == "descend" and again.transitions[0].reward == 0.0
    assert path.read_bytes() == original


@pytest.mark.parametrize("location", ["top", "episode"])
@pytest.mark.parametrize("role", ["probe_train", "probe_test", "final_test", "inner_guard"])
def test_should_refuse_non_train_roles_at_each_qualification_layer(tmp_path, role, location):
    raw = receipt()
    (raw if location == "top" else raw["candidate_episode"])["role"] = role
    with pytest.raises(ValueError):
        cli._read_readiness_teacher(write(tmp_path / "role.json", raw), 17)


@pytest.mark.parametrize(
    "path,value",
    [
        (("qualified",), False),
        (("qualified",), 1),
        (("teacher_definition",), MISSING),
        (("teacher_definition",), "pick_cube_fixed_schedule_v1"),
        (("observed_success",), False),
        (("final_success",), False),
        (("error",), "paid native failure"),
        (("traceback",), "retained traceback"),
        (("seed",), 19),
        (("seed",), 17.0),
        (("context",), "pick-cube-train-19"),
        (("candidate_episode", "episode_id"), "pick-cube-task-teacher-17"),
        (("candidate_episode", "context"), "other"),
        (("task",), "other"),
        (("controller",), "pd_joint_pos"),
        (("creation_attempts",), True),
        (("geometry_attempts",), 3),
        (("geometry_returns",), 2.0),
        (("teacher_actions",), 1),
        (("stages",), ["descend", 1]),
        (("stages",), ["descend", "align"]),
        (("stages",), MISSING),
        (("work",), MISSING),
        (("work", "completed_steps"), 1),
        (("work", "completed_control_physics_steps"), 9),
        (("candidate_episode", "transitions", 1, "action"), [0.1, 0, 0, -1]),
        (("candidate_episode", "transitions", 1, "reward"), 0),
        (("candidate_episode", "transitions", 0, "reward"), 0.5),
        (("guard_seed",), 31),
        (("labels_arrived_at",), 2),
        (("camera_context",), {}),
        (("mass_work",), {}),
        (("lighting_readback",), {}),
        (("candidate_episode", "spec", "observation_dim"), 21),
        (("candidate_episode", "spec", "action_units"), ["m"] * 4),
        (("candidate_episode", "spec", "upper_bounds"), [2.0] * 4),
        (("candidate_episode", "transitions", 1, "terminated"), False),
        (("candidate_episode", "transitions", 0, "next_observation", 0), 99.0),
        (("optimizer_updates",), True),
        (("provenance",), "provisional_geometry_readiness_v1"),
    ],
)
def test_should_refuse_incomplete_relabelled_or_corrupted_readiness_receipts(tmp_path, path, value):
    raw = receipt()
    change(raw, path, value)
    with pytest.raises(ValueError):
        cli._read_readiness_teacher(write(tmp_path / "bad.json", raw), 17)


@pytest.mark.parametrize(
    "fault",
    [
        "recipe_missing",
        "recipe_wrong_type",
        "visual_cost",
        "exports",
        "rgb_terminal",
        "mode_mixing",
    ],
)
def test_should_bind_original_rgb_recipe_range_and_complete_encoding_costs(tmp_path, fault):
    raw = receipt(visual=True)
    if fault == "recipe_missing":
        del raw["observation_recipe"]["feature_dim"]
    elif fault == "recipe_wrong_type":
        raw["observation_recipe"]["feature_dim"] = 789.0
    elif fault == "visual_cost":
        raw["visual_work"]["returned_feature_values"] = 2366
    elif fault == "exports":
        raw["visual_work"]["frame_export_copies"] = 1
    elif fault == "rgb_terminal":
        raw["candidate_episode"]["transitions"][-1]["next_observation"][-1] = 1.01
    with pytest.raises(ValueError):
        cli._read_readiness_teacher(
            write(tmp_path / "rgb.json", raw), 17, visual_observation=fault != "mode_mixing"
        )


@pytest.mark.parametrize("visual", [1, None, "true"])
def test_should_reject_non_boolean_visual_mode_before_file_access(visual):
    with pytest.raises(ValueError):
        cli._read_readiness_teacher(Path("missing-readiness-file"), 17, visual_observation=visual)


@pytest.mark.parametrize("seed", [True, 17.0, 23, 31])
def test_should_reject_non_train_seed_before_file_access(seed):
    with pytest.raises(ValueError):
        cli._read_readiness_teacher(Path("missing-readiness-file"), seed)


@pytest.mark.parametrize("fault", ["duplicate_top", "duplicate_nested", "nonfinite", "utf8"])
def test_should_refuse_ambiguous_or_nonfinite_serialized_receipts(tmp_path, fault):
    encoded = json.dumps(receipt(), separators=(",", ":"))
    if fault == "duplicate_top":
        encoded = encoded.replace('"qualified":true', '"qualified":false,"qualified":true')
    elif fault == "duplicate_nested":
        encoded = encoded.replace('"completed_steps":2', '"completed_steps":1,"completed_steps":2')
    elif fault == "nonfinite":
        encoded = encoded.replace('"reward":0.0', '"reward":NaN', 1)
    path = tmp_path / "ambiguous.json"
    path.write_bytes(b"\xff" if fault == "utf8" else encoded.encode())
    with pytest.raises(ValueError):
        cli._read_readiness_teacher(path, 17)


@pytest.mark.parametrize("visual", [False, True])
def test_should_read_one_owned_capped_payload_before_decoding(visual):
    limit = (4 if visual else 1) * 1024**2
    reads, opens = [], []

    class Handle(io.BytesIO):
        def read(self, size=-1):
            reads.append(size)
            return super().read(size)

    class Input:
        def open(self, mode):
            opens.append(mode)
            return Handle(b" " * (limit + 1))

    source: Any = Input()
    with pytest.raises(ValueError, match="MiB"):
        cli._read_readiness_teacher(source, 17, visual_observation=visual)
    assert opens == ["rb"] and reads == [limit + 1]


@pytest.mark.parametrize("relabel", [False, True])
def test_should_require_explicit_readiness_route_even_with_a_legacy_identity(tmp_path, relabel):
    raw = receipt()
    if relabel:
        raw["candidate_episode"]["episode_id"] = "pick-cube-task-teacher-17"
    with pytest.raises(ValueError, match="readiness"):
        cli._read_teacher(write(tmp_path / "readiness.json", raw), 17)


@pytest.fixture
def stub_runners(monkeypatch):
    for name in (
        "run_demo",
        "run_coupled_demo",
        "run_matched_demo",
        "run_rejection_demo",
        "run_continuous_demo",
        "run_continuous_learnability_demo",
    ):
        monkeypatch.setattr(cli, name, forbidden)
    monkeypatch.setattr(robot, "run_learning_rollout", forbidden)


@pytest.mark.parametrize(
    "scenario,flags",
    [
        ("continuous-nearest", []),
        (
            "continuous-mlp-learnability",
            ["--phase-updates", "1", "--a-only-control", "--binary-gripper-axis", "3"],
        ),
        ("continuous-mlp-learnability", ["--visual-observation", "--public-vector-only"]),
    ],
)
def test_should_route_both_readiness_seeds_and_preserve_explicit_learning_options(
    stub_runners, monkeypatch, tmp_path, capsys, scenario, flags
):
    visual = "--visual-observation" in flags
    paths = [write(tmp_path / f"{seed}.json", receipt(seed, visual=visual)) for seed in (17, 19)]
    calls = []

    def run(warm, new, **options):
        calls.append((warm, new, options))
        return dict(complete=True, evidence="stubbed")

    monkeypatch.setattr(
        cli,
        "run_continuous_demo"
        if scenario == "continuous-nearest"
        else "run_continuous_learnability_demo",
        run,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stream",
            "--scenario",
            scenario,
            "--warm-teacher",
            str(paths[0]),
            "--new-teacher",
            str(paths[1]),
            "--readiness-teachers",
            *flags,
        ],
    )
    cli.main()
    output = json.loads(capsys.readouterr().out)
    warm, new, options = calls[0]
    assert warm.episode_id.endswith("17") and new.episode_id.endswith("19")
    assert warm.episode_id.startswith("pick-cube-readiness-teacher-")
    assert new.context == "pick-cube-train-19"
    assert output["input_teachers"]["warm"]["teacher_definition"] == DEFINITION
    assert output["input_teachers"]["new"]["teacher_definition"] == DEFINITION
    if "--phase-updates" in flags:
        assert options == dict(phase_updates=1, a_only_control=True, binary_gripper_axis=3)
    else:
        assert options == {}
    if "--public-vector-only" in flags:
        assert warm.spec.observation_dim == new.spec.observation_dim == 21
        assert output["input_projection"]["observation_projections"] == 8


@pytest.mark.parametrize(
    "flags",
    [
        ["--scenario", "stream"],
        ["--scenario", "continuous-nearest", "--mass-drift"],
        ["--scenario", "continuous-nearest", "--visual-observation", "--camera-drift"],
        ["--scenario", "continuous-nearest", "--visual-observation", "--lighting-drift"],
    ],
)
def test_should_refuse_incompatible_readiness_cli_flags_before_any_reader(
    stub_runners, monkeypatch, flags
):
    monkeypatch.setattr(cli, "_read_teacher", forbidden)
    monkeypatch.setattr(cli, "_read_readiness_teacher", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stream",
            "--readiness-teachers",
            "--warm-teacher",
            "missing-A",
            "--new-teacher",
            "missing-B",
            *flags,
        ],
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


@pytest.mark.parametrize("visual", [False, True])
def test_should_route_robot_cli_readiness_inputs_without_relabelling_or_metadata_loss(
    stub_runners, monkeypatch, tmp_path, capsys, visual
):
    paths = [write(tmp_path / f"{seed}.json", receipt(seed, visual=visual)) for seed in (17, 19)]
    calls = []

    def run(output, warm, new, **options):
        calls.append((output, warm, new, options))
        return dict(complete=False, training={"paid": 1}, rollout=None)

    monkeypatch.setattr(robot, "run_learning_rollout", run)
    flags = ["--visual-observation"] if visual else []
    code = robot.main(
        [
            "--output",
            str(tmp_path / "combined.json"),
            "--warm-teacher",
            str(paths[0]),
            "--new-teacher",
            str(paths[1]),
            "--readiness-teachers",
            "--phase-updates",
            "1",
            *flags,
        ]
    )
    assert code == 1
    assert json.loads(capsys.readouterr().out)["training"] == {"paid": 1}
    _, warm, new, options = calls[0]
    assert (
        warm.episode_id == "pick-cube-readiness-teacher-17"
        and new.episode_id == "pick-cube-readiness-teacher-19"
    )
    assert options == dict(learner_family="bp", phase_updates=1, visual_observation=visual)


def test_should_refuse_robot_lighting_readiness_mix_before_input_or_output(
    stub_runners, monkeypatch, tmp_path
):
    monkeypatch.setattr(robot, "_read_teacher", forbidden)
    monkeypatch.setattr(robot, "_read_readiness_teacher", forbidden)
    output = tmp_path / "absent.json"
    with pytest.raises(SystemExit) as error:
        robot.main(
            [
                "--output",
                str(output),
                "--warm-teacher",
                "missing-A",
                "--new-teacher",
                "missing-B",
                "--readiness-teachers",
                "--lighting-drift",
                "--visual-observation",
            ]
        )
    assert error.value.code == 2 and not output.exists()


def test_should_leave_default_stream_dispatch_and_report_schema_unchanged(
    stub_runners, monkeypatch, capsys
):
    values = [object(), object()]
    reads, runs = [], []

    def read(path, seed, **options):
        reads.append((path, seed, options))
        return values[len(reads) - 1]

    def run(*args, **options):
        runs.append((args, options))
        return dict(complete=True, legacy="unchanged")

    monkeypatch.setattr(cli, "_read_teacher", read)
    monkeypatch.setattr(cli, "_read_readiness_teacher", forbidden)
    monkeypatch.setattr(cli, "run_continuous_learnability_demo", run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stream",
            "--scenario",
            "continuous-mlp-learnability",
            "--warm-teacher",
            "A.json",
            "--new-teacher",
            "B.json",
        ],
    )
    cli.main()
    assert [(seed, options) for _, seed, options in reads] == [(17, {}), (19, {})]
    assert runs == [((values[0], values[1]), {})]
    assert json.loads(capsys.readouterr().out) == dict(complete=True, legacy="unchanged")


def test_should_print_failed_paid_readiness_run_without_claiming_completion(
    stub_runners, monkeypatch, tmp_path, capsys
):
    paths = [write(tmp_path / f"{seed}.json", receipt(seed)) for seed in (17, 19)]
    returned = dict(
        complete=False,
        applied_updates=[{"diagnostic": {"value": 0.25}}],
        progress={"updates_completed": 1},
    )
    monkeypatch.setattr(cli, "run_continuous_demo", lambda *args, **kwargs: returned)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stream",
            "--scenario",
            "continuous-nearest",
            "--warm-teacher",
            str(paths[0]),
            "--new-teacher",
            str(paths[1]),
            "--readiness-teachers",
        ],
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 1
    observed = json.loads(capsys.readouterr().out)
    assert observed["complete"] is False and observed["progress"]["updates_completed"] == 1
    assert observed["applied_updates"] == returned["applied_updates"]
    assert observed["input_teachers"]["warm"]["teacher_definition"] == DEFINITION


@pytest.fixture(scope="module")
def learned(tmp_path_factory):
    folder = tmp_path_factory.mktemp("readiness-learned-return")
    paths = [write(folder / f"{seed}.json", receipt(seed)) for seed in (17, 19)]
    output = folder / "combined.json"
    captures: dict[str, Any] = {}
    original = robot.run_continuous_learnability
    environment = StateEnvironment()

    def train(*args, **kwargs):
        result = original(*args, **kwargs)
        captures["result"] = result
        assert result.candidate is not None
        captures["before"] = result.candidate.snapshot_state()
        return result

    def factory(**options):
        assert options == {}
        return environment

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(robot, "run_continuous_learnability", train)
        patch.setattr(native, "_make_environment", factory)
        patch.setattr(smoke, "version", lambda _: "fake-readiness-learning")
        code = robot.main(
            [
                "--output",
                str(output),
                "--warm-teacher",
                str(paths[0]),
                "--new-teacher",
                str(paths[1]),
                "--readiness-teachers",
                "--phase-updates",
                "1",
            ]
        )
    return code, json.loads(output.read_text(encoding="utf8")), captures, environment


def test_should_train_from_admitted_readiness_ids_and_return_actual_detached_candidate_to_robot(
    learned,
):
    code, report, _, environment = learned
    assert code == 0 and report["complete"] is True
    assert (
        report["training"]["source_episodes"]["warm"]["episode_id"]
        == "pick-cube-readiness-teacher-17"
    )
    assert (
        report["training"]["source_episodes"]["new"]["episode_id"]
        == "pick-cube-readiness-teacher-19"
    )
    assert report["training"]["progress"]["updates_completed"] == 3
    assert (
        report["rollout"]["source_updates"] == 3 and report["rollout"]["source_training_rows"] == 6
    )
    assert report["rollout"]["seed"] == 17 and report["rollout"]["context"] == "pick-cube"
    assert (
        environment.reset_calls == [17]
        and len(environment.commands) == 2
        and environment.close_calls == 1
    )


def test_should_keep_candidate_full_parameters_velocity_and_counters_frozen_through_return(learned):
    _, report, captures, _ = learned
    before, after = captures["before"], captures["result"].candidate.snapshot_state()
    assert (
        before.policy == after.policy and before.completed_updates == after.completed_updates == 3
    )
    assert before.training_examples == after.training_examples == 6
    assert all(
        left.tolist() == right.tolist()
        for left, right in zip(
            before.parameters + before.velocity, after.parameters + after.velocity, strict=True
        )
    )
    assert report["candidate_unchanged"] is True
    assert report["rollout"]["joint_position_exact"] is True
    assert report["rollout"]["joint_position_validation_samples"] == 3
    assert report["rollout"]["optimizer_updates"] == 0
