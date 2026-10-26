"""Fixed-camera TRAIN receipt parsing and CLI composition, with stubbed learners.

No model/fit/RNG/optimizer/prediction/native work. Two real public-vector helper
calls validate/project16 observations:12,624 raw values,336 projected values,
2,688 projected numeric bytes. Per platform:66 bounded4MiB+1 reader calls,
4 fixture identity reads,5 stub learner calls;69 expanded cases.
"""

from copy import deepcopy
from dataclasses import asdict
import io
import json
from pathlib import Path
import sys

import pytest

from scripts import run_native_stream_demo as cli
from src.core.pick_cube_camera_context import camera_context_recipe, pick_cube_camera_context
from src.core.pick_cube_rgb_features import pick_cube_rgb_recipe

LIMIT = 4 * 1024**2
MISSING = object()


def receipt(*, camera: bool = True, seed: int = 17, optimizer: bool = True):
    config = pick_cube_camera_context()
    context = config.context if camera else "pick-cube" if seed == 17 else "pick-cube-train-19"
    identity = f"{config.context}-teacher" if camera else f"pick-cube-task-teacher-{seed}"
    spec = {
        "observation_dim": 789,
        "action_units": [
            "normalized_root_dx",
            "normalized_root_dy",
            "normalized_root_dz",
            "normalized_gripper",
        ],
        "lower_bounds": [-1.0] * 4,
        "upper_bounds": [1.0] * 4,
    }
    observations = [[float(index)] + [0.0] * 20 + [0.25] * 768 for index in range(3)]
    raw = {
        "qualified": True,
        "observed_success": True,
        "final_success": True,
        "seed": seed,
        "role": "train",
        "context": context,
        "task": "PickCube-v1",
        "controller": "pd_ee_delta_pos",
        "creation_attempts": 1,
        "geometry_attempts": 2,
        "geometry_returns": 2,
        "teacher_actions": 2,
        "stages": ["align", "align"],
        "work": {
            "step_attempts": 2,
            "completed_steps": 2,
            "completed_control_physics_steps": 10,
            "possible_failed_physics_steps": 0,
        },
        "error": None,
        "traceback": None,
        "guard_seed": None,
        "labels_arrived_at": None,
        "camera_context": asdict(config) if camera else None,
        "mass_context": None,
        "mass_readback": None,
        "mass_work": None,
        "lighting_context": None,
        "lighting_readback": None,
        "lighting_work": None,
        "observation_recipe": asdict(
            camera_context_recipe(config) if camera else pick_cube_rgb_recipe()
        ),
        "visual_work": {
            "observation_attempts": 3,
            "observation_returns": 3,
            "input_frame_bytes_upper_bound": 147456,
            "feature_values_upper_bound": 2367,
            "returned_frame_bytes": 147456,
            "returned_feature_values": 2367,
            "returned_feature_numeric_bytes": 18936,
            "current_owned_frame_bytes": 49152,
            "frame_export_copies": 0,
            "frame_export_bytes": 0,
        },
        "candidate_episode": {
            "episode_id": identity,
            "context": context,
            "role": "train",
            "spec": spec,
            "transitions": [
                {
                    "step": index,
                    "observation": observations[index],
                    "action": [0.5, 0.0, 1.0, 1.0],
                    "reward": float(index == 1),
                    "next_observation": observations[index + 1],
                    "terminated": index == 1,
                    "truncated": False,
                }
                for index in range(2)
            ],
        },
    }
    if optimizer:
        raw["optimizer_updates"] = 0
    return json.loads(json.dumps(raw))


def write(path: Path, raw) -> Path:
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


@pytest.fixture(autouse=True)
def forbid_runners(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("input parsing/routing executed a learner runner")

    for name in (
        "run_demo",
        "run_coupled_demo",
        "run_matched_demo",
        "run_rejection_demo",
        "run_continuous_demo",
        "run_continuous_learnability_demo",
    ):
        monkeypatch.setattr(cli, name, forbidden)


@pytest.mark.parametrize("optimizer", [False, True])
def test_should_release_owned_camera_pairs_and_original_paid_acquisition(tmp_path, optimizer):
    raw = receipt(optimizer=optimizer)
    path = write(tmp_path / "camera.json", raw)
    original = path.read_bytes()
    episode, metadata = cli._read_camera_teacher(path, visual_observation=True)
    expected = {
        "mode": "camera_drift",
        "warm_seed": 17,
        "new_seed": 17,
        "warm_context": "pick-cube",
        "new_context": raw["context"],
        "camera_context": raw["camera_context"],
        "teacher_work": raw["work"],
        "teacher_visual_work": raw["visual_work"],
        "teacher_creation_attempts": 1,
        "teacher_geometry_attempts": 2,
        "teacher_geometry_returns": 2,
        "teacher_actions": 2,
        "source_file_bytes": len(original),
        "per_file_byte_limit": LIMIT,
        "acquisition_scope": "retained_teacher_work_not_new_simulator_work",
    }
    assert json.loads(json.dumps(metadata)) == expected
    assert episode.role == "train" and episode.spec.observation_dim == 789
    assert episode.episode_id == raw["candidate_episode"]["episode_id"]
    assert episode.context == raw["context"] and episode.transitions[-1].reward == 1.0
    metadata["camera_context"]["seed"] = 19
    metadata["teacher_visual_work"]["returned_feature_values"] = 0
    object.__setattr__(episode.transitions[0], "observation", (999.0,) * 789)
    repeated, again = cli._read_camera_teacher(path, visual_observation=True)
    assert again["camera_context"]["seed"] == 17
    assert again["teacher_visual_work"]["returned_feature_values"] == 2367
    assert repeated.transitions[0].observation[0] == 0.0
    assert path.read_bytes() == original


@pytest.mark.parametrize("role", ["inner_guard", "probe_train", "probe_test", "final_test"])
@pytest.mark.parametrize("location", ["top", "episode"])
def test_should_refuse_heldout_or_guard_roles_without_relabelling(tmp_path, role, location):
    raw = receipt()
    raw["role"] = role if location == "top" else "train"
    if location == "episode":
        raw["candidate_episode"]["role"] = role
    with pytest.raises(ValueError):
        cli._read_camera_teacher(write(tmp_path / "role.json", raw), visual_observation=True)


@pytest.mark.parametrize(
    "path,value",
    [
        (("qualified",), False),
        (("seed",), 17.0),
        (("context",), "pick-cube"),
        (("candidate_episode", "episode_id"), "pick-cube-task-teacher-17"),
        (("controller",), "pd_joint_pos"),
        (("observed_success",), False),
        (("final_success",), False),
        (("error",), "native failure"),
        (("traceback",), "partial failure"),
        (("guard_seed",), 31),
        (("labels_arrived_at",), 3),
        (("mass_context",), {"seed": 17}),
        (("lighting_context",), {"seed": 17}),
        (("creation_attempts",), True),
        (("geometry_returns",), 2.0),
        (("teacher_actions",), 1),
        (("work", "completed_control_physics_steps"), 9),
        (("stages",), ["descend", "align"]),
        (("visual_work", "observation_returns"), 2),
        (("visual_work", "returned_feature_values"), 2367.0),
        (("visual_work", "frame_export_copies"), 1),
        (("optimizer_updates",), False),
        (("observation_recipe", "camera_eye", 1), 0.0),
        (("observation_recipe", "feature_dim"), 789.0),
        (("observation_recipe", "pixel_divisor"), 255),
        (("camera_context", "seed"), 17.0),
    ],
)
def test_should_refuse_malformed_binding_recipe_or_paid_acquisition_before_runner(
    tmp_path, path, value
):
    raw = receipt()
    change(raw, path, value)
    with pytest.raises(ValueError):
        cli._read_camera_teacher(write(tmp_path / "invalid.json", raw), visual_observation=True)


@pytest.mark.parametrize("value", [MISSING, None, {}])
def test_should_refuse_missing_camera_declaration_without_backfilling(tmp_path, value):
    raw = receipt()
    change(raw, ("camera_context",), value)
    with pytest.raises(ValueError):
        cli._read_camera_teacher(write(tmp_path / "missing.json", raw), visual_observation=True)


@pytest.mark.parametrize("field", ["context", "seed"])
def test_should_not_backfill_missing_recorded_camera_fields(tmp_path, field):
    raw = receipt()
    del raw["camera_context"][field]
    with pytest.raises(ValueError):
        cli._read_camera_teacher(
            write(tmp_path / "partial-context.json", raw), visual_observation=True
        )


@pytest.mark.parametrize("damage", ["terminal_pixel", "seam"])
def test_should_validate_entire_source_including_terminal_pixels_and_seams(tmp_path, damage):
    raw = receipt()
    row = raw["candidate_episode"]["transitions"][-1]
    if damage == "terminal_pixel":
        row["next_observation"][21] = 1.01
    else:
        row["observation"][0] = 99.0
    with pytest.raises(ValueError):
        cli._read_camera_teacher(write(tmp_path / "trace.json", raw), visual_observation=True)


@pytest.mark.parametrize("flag", [False, 1, None])
def test_should_refuse_nonvisual_request_before_open(monkeypatch, flag):
    monkeypatch.setattr(
        Path, "open", lambda *args, **kwargs: pytest.fail("invalid visual flag opened a file")
    )
    with pytest.raises(ValueError, match="visual-observation"):
        cli._read_camera_teacher(Path("never.json"), visual_observation=flag)


@pytest.mark.parametrize("where", ["qualification", "recipe"])
def test_should_refuse_duplicate_json_before_overwriting_authority(tmp_path, where):
    encoded = json.dumps(receipt())
    if where == "qualification":
        encoded = '{"qualified": false,' + encoded[1:]
    else:
        encoded = encoded.replace('"feature_dim": 789', '"feature_dim": 789, "feature_dim": 789', 1)
    path = tmp_path / "duplicate.json"
    path.write_text(encoded, encoding="utf8")
    with pytest.raises(ValueError, match="duplicate"):
        cli._read_camera_teacher(path, visual_observation=True)


def test_should_refuse_nonfinite_json_before_training(tmp_path):
    raw = receipt()
    raw["candidate_episode"]["transitions"][-1]["next_observation"][0] = float("nan")
    with pytest.raises(ValueError):
        cli._read_camera_teacher(write(tmp_path / "nonfinite.json", raw), visual_observation=True)


@pytest.mark.parametrize("oversized", [False, True])
def test_should_read_one_owned_bounded_payload(monkeypatch, oversized):
    payload = b" " * (LIMIT + 1) if oversized else json.dumps(receipt()).encode("utf8")
    requests: list[int] = []

    class Buffer(io.BytesIO):
        def read(self, size=-1):
            requests.append(size)
            return super().read(size)

    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: Buffer(payload))
    if oversized:
        with pytest.raises(ValueError, match="4MiB"):
            cli._read_camera_teacher(Path("owned.json"), visual_observation=True)
    else:
        episode, metadata = cli._read_camera_teacher(Path("owned.json"), visual_observation=True)
        assert len(episode.transitions) == 2 and metadata["source_file_bytes"] == len(payload)
    assert requests == [LIMIT + 1]


@pytest.mark.parametrize("reader", ["default", "mass", "lighting"])
def test_should_refuse_camera_receipt_on_other_reader_routes(tmp_path, reader):
    path = write(tmp_path / "camera.json", receipt())
    with pytest.raises(ValueError):
        if reader == "default":
            cli._read_teacher(path, 17, visual_observation=True)
        else:
            getattr(cli, f"_read_{reader}_teacher")(path, visual_observation=True)


def test_should_keep_default_none_camera_metadata_accepted(tmp_path):
    raw = receipt(camera=False)
    actual = cli._read_teacher(write(tmp_path / "default.json", raw), 17, visual_observation=True)
    assert actual.context == "pick-cube" and actual.episode_id == "pick-cube-task-teacher-17"


def test_should_project_default_A_and_camera_B_with_two_truthful_source_recipes(tmp_path):
    warm = cli._read_teacher(
        write(tmp_path / "a.json", receipt(camera=False)), 17, visual_observation=True
    )
    new, _ = cli._read_camera_teacher(
        write(tmp_path / "b.json", receipt()), visual_observation=True
    )
    originals = (asdict(warm), asdict(new))
    first, second, work = cli._project_public_inputs(
        warm, new, camera_context=pick_cube_camera_context()
    )
    assert first.spec.observation_dim == second.spec.observation_dim == 21
    assert first.context == "pick-cube" and second.context == pick_cube_camera_context().context
    assert first.transitions[0].observation == warm.transitions[0].observation[:21]
    assert second.transitions[-1].next_observation == new.transitions[-1].next_observation[:21]
    assert (asdict(warm), asdict(new)) == originals
    assert work["source_recipe"] == asdict(pick_cube_rgb_recipe())
    assert work["new_source_recipe"] == asdict(camera_context_recipe(pick_cube_camera_context()))
    assert work["observation_projections"] == 8
    assert work["source_values_validated"] == 6312
    assert work["projected_values"] == 168 and work["projected_numeric_bytes"] == 1344


@pytest.fixture
def cli_paths(tmp_path):
    return (
        write(tmp_path / "a.json", receipt(camera=False)),
        write(tmp_path / "camera.json", receipt()),
    )


def argv(paths, *, scenario="continuous-nearest", extra=()):
    return [
        "stream",
        "--scenario",
        scenario,
        "--visual-observation",
        "--camera-drift",
        "--warm-teacher",
        str(paths[0]),
        "--new-teacher",
        str(paths[1]),
        *extra,
    ]


def test_should_route_actual_camera_receipts_to_nearest_stub_and_report_both_recipes(
    cli_paths, monkeypatch, capsys
):
    calls = []

    def run(warm, new):
        calls.append((warm, new))
        return {"complete": True}

    monkeypatch.setattr(cli, "run_continuous_demo", run)
    monkeypatch.setattr(sys, "argv", argv(cli_paths))
    cli.main()
    report = json.loads(capsys.readouterr().out)
    assert len(calls) == 1
    assert (
        calls[0][0].context == "pick-cube"
        and calls[0][1].context == pick_cube_camera_context().context
    )
    assert report["input_context"]["mode"] == "camera_drift"
    assert report["input_context"]["new_seed"] == 17
    assert report["input_observation"] == {
        "mode": "public_rgb_camera_drift",
        "warm_recipe": json.loads(json.dumps(asdict(pick_cube_rgb_recipe()))),
        "new_recipe": json.loads(
            json.dumps(asdict(camera_context_recipe(pick_cube_camera_context())))
        ),
        "per_file_byte_limit": LIMIT,
    }


def test_should_compose_camera_public21_with_real_projection_and_no_model(
    cli_paths, monkeypatch, capsys
):
    calls = []

    def run(warm, new):
        calls.append((warm, new))
        return {"complete": True}

    monkeypatch.setattr(cli, "run_continuous_learnability_demo", run)
    monkeypatch.setattr(
        sys,
        "argv",
        argv(cli_paths, scenario="continuous-mlp-learnability", extra=("--public-vector-only",)),
    )
    cli.main()
    report = json.loads(capsys.readouterr().out)
    assert calls[0][0].spec.observation_dim == calls[0][1].spec.observation_dim == 21
    assert report["input_observation"]["source_mode"] == "public_rgb_camera_drift"
    assert report["input_projection"]["source_recipe"] == report["input_observation"]["warm_recipe"]
    assert (
        report["input_projection"]["new_source_recipe"] == report["input_observation"]["new_recipe"]
    )


@pytest.mark.parametrize(
    "extra,expected",
    [
        (
            ("--learner-family", "pc", "--phase-updates", "2", "--a-only-control"),
            {"learner_family": "pc", "phase_updates": 2, "a_only_control": True},
        ),
        (
            ("--binary-gripper-axis", "3", "--phase-updates", "1"),
            {"binary_gripper_axis": 3, "phase_updates": 1},
        ),
    ],
)
def test_should_forward_existing_model_options_without_changing_camera_inputs(
    cli_paths, monkeypatch, capsys, extra, expected
):
    calls = []

    def run(warm, new, **options):
        calls.append((warm, new, options))
        return {"complete": True}

    monkeypatch.setattr(cli, "run_continuous_learnability_demo", run)
    monkeypatch.setattr(
        sys, "argv", argv(cli_paths, scenario="continuous-mlp-learnability", extra=extra)
    )
    cli.main()
    assert calls[0][2] == expected
    assert calls[0][1].context == pick_cube_camera_context().context
    assert json.loads(capsys.readouterr().out)["complete"] is True


def test_should_keep_partial_paid_runner_output_and_exit_one(cli_paths, monkeypatch, capsys):
    paid = {
        "complete": False,
        "progress": {"updates_completed": 1},
        "error": "synthetic paid failure",
    }
    monkeypatch.setattr(cli, "run_continuous_demo", lambda warm, new: deepcopy(paid))
    monkeypatch.setattr(sys, "argv", argv(cli_paths))
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 1
    output = json.loads(capsys.readouterr().out)
    assert output["progress"] == paid["progress"] and output["error"] == paid["error"]
    assert output["input_context"]["mode"] == "camera_drift"


@pytest.mark.parametrize(
    "arguments",
    [
        ["--camera-drift"],
        [
            "--scenario",
            "continuous-nearest",
            "--camera-drift",
            "--warm-teacher",
            "a",
            "--new-teacher",
            "b",
        ],
        [
            "--scenario",
            "continuous-nearest",
            "--visual-observation",
            "--camera-drift",
            "--mass-drift",
            "--warm-teacher",
            "a",
            "--new-teacher",
            "b",
        ],
        [
            "--scenario",
            "continuous-nearest",
            "--visual-observation",
            "--camera-drift",
            "--lighting-drift",
            "--warm-teacher",
            "a",
            "--new-teacher",
            "b",
        ],
        [
            "--scenario",
            "continuous-nearest",
            "--visual-observation",
            "--camera-drift",
            "--seed",
            "17",
            "--warm-teacher",
            "a",
            "--new-teacher",
            "b",
        ],
        ["--scenario", "continuous-nearest", "--visual-observation", "--camera-drift"],
        [
            "--scenario",
            "continuous-nearest",
            "--visual-observation",
            "--camera-drift",
            "--public-vector-only",
            "--warm-teacher",
            "a",
            "--new-teacher",
            "b",
        ],
        [
            "--scenario",
            "continuous-mlp-learnability",
            "--visual-observation",
            "--camera-drift",
            "--public-vector-only",
            "--rgb-range-normalization",
            "--warm-teacher",
            "a",
            "--new-teacher",
            "b",
        ],
    ],
)
def test_should_refuse_invalid_camera_cli_combinations_before_any_file_io(monkeypatch, arguments):
    monkeypatch.setattr(
        Path, "open", lambda *args, **kwargs: pytest.fail("invalid CLI opened teacher")
    )
    monkeypatch.setattr(sys, "argv", ["stream", *arguments])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
