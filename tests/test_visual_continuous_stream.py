"""Strict declared visual receipts and synthetic delayed-stream behavior; no camera work."""

from dataclasses import asdict
import io
import json
import sys

import pytest

from scripts import run_native_stream_demo as cli
from src.adapters import continuous_stream_demo as demo
from src.core.pick_cube_rgb_features import pick_cube_rgb_recipe
from src.infra.maniskill_continuous import ACTION_UNITS


def recipe():
    return json.loads(json.dumps(asdict(pick_cube_rgb_recipe())))


def teacher(seed=17, *, visual=True):
    context = "pick-cube" if seed == 17 else "pick-cube-train-19"
    dimension = 789 if visual else 42
    offset = 0.0 if seed == 17 else 10.0
    observations = [[offset + index] + [0.0] * (dimension - 1) for index in range(3)]
    controls = (
        [[-1.0, 0.0, 0.0, 1.0], [-0.5, 0.0, 0.0, -1.0]]
        if seed == 17
        else [
            [1.0, 0.0, 0.0, 1.0],
            [0.5, 0.0, 0.0, -1.0],
        ]
    )
    return dict(
        fixture_kind="synthetic_not_an_sdk_trajectory",
        qualified=True,
        final_success=True,
        error=None,
        role="train",
        seed=seed,
        context=context,
        task="PickCube-v1",
        controller="pd_ee_delta_pos",
        observation_recipe=recipe() if visual else None,
        candidate_episode=dict(
            episode_id=f"pick-cube-task-teacher-{seed}",
            context=context,
            role="train",
            spec=dict(
                observation_dim=dimension,
                action_units=list(ACTION_UNITS),
                lower_bounds=[-1.0] * 4,
                upper_bounds=[1.0] * 4,
            ),
            transitions=[
                dict(
                    step=index,
                    observation=observations[index].copy(),
                    action=controls[index].copy(),
                    reward=float(index == 1),
                    next_observation=observations[index + 1].copy(),
                    terminated=index == 1,
                    truncated=False,
                )
                for index in range(2)
            ],
        ),
    )


def write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf8")
    return path


@pytest.mark.parametrize("seed", [17, 19])
def test_should_read_owned_visual_records_with_exact_seed_context_recipe_and_immutable_source(
    tmp_path, seed
):
    raw = teacher(seed)
    path = write(tmp_path / "visual.json", raw)
    before = path.read_bytes()
    episode = cli._read_teacher(path, seed, visual_observation=True)
    assert episode.spec.observation_dim == 789 and episode.spec.action_units == ACTION_UNITS
    assert episode.role == "train" and episode.context == raw["context"]
    assert episode.episode_id == raw["candidate_episode"]["episode_id"]
    assert type(episode.transitions[0].observation) is tuple
    assert episode.transitions[0].observation == tuple(
        raw["candidate_episode"]["transitions"][0]["observation"]
    )
    object.__setattr__(episode.transitions[0], "observation", (999.0,) * 789)
    repeated = cli._read_teacher(path, seed, visual_observation=True)
    assert repeated.transitions[0].observation[0] == (0.0 if seed == 17 else 10.0)
    assert path.read_bytes() == before


def test_should_accept_visual_receipt_larger_than_one_mib_and_preserve_state_read_limit(tmp_path):
    encoded = json.dumps(teacher()).encode("utf8")
    path = tmp_path / "large-visual.json"
    path.write_bytes(encoded + b" " * (1024**2 + 7 - len(encoded)))
    assert 1024**2 < path.stat().st_size <= 4 * 1024**2
    assert cli._read_teacher(path, 17, visual_observation=True).spec.observation_dim == 789
    with pytest.raises(ValueError, match="1MiB"):
        cli._read_teacher(path, 17)


def test_should_refuse_visual_input_beyond_four_mib_before_json_parsing(tmp_path):
    path = tmp_path / "too-large.json"
    path.write_bytes(b" " * (4 * 1024**2 + 1))
    with pytest.raises(ValueError, match="4MiB"):
        cli._read_teacher(path, 17, visual_observation=True)


@pytest.mark.parametrize("visual", [False, True])
def test_should_request_only_one_bounded_read_in_the_selected_mode(visual):
    payload = json.dumps(teacher(visual=visual)).encode("utf8")
    sizes = []
    expected_cap = (4 if visual else 1) * 1024**2

    class BoundedReader(io.BytesIO):
        def read(self, size=-1):
            sizes.append(size)
            assert size == expected_cap + 1
            return super().read(size)

    class ReceiptPath:
        def open(self, mode):
            assert mode == "rb"
            return BoundedReader(payload)

    assert cli._read_teacher(ReceiptPath(), 17, visual_observation=visual).spec.observation_dim == (
        789 if visual else 42
    )
    assert sizes == [expected_cap + 1]


@pytest.mark.parametrize("bad", [0, 1, None, "True", []])
def test_should_refuse_non_boolean_mode_before_opening_any_source(bad):
    class ForbiddenPath:
        def open(self, *_):
            pytest.fail("invalid visual mode reached file IO")

    with pytest.raises(ValueError):
        cli._read_teacher(ForbiddenPath(), 17, visual_observation=bad)


@pytest.mark.parametrize(
    "field",
    [
        "recipe_id",
        "camera",
        "frame_shape",
        "camera_eye",
        "camera_target",
        "camera_fov",
        "camera_near",
        "camera_far",
        "shader_pack",
        "render_backend",
        "pixel_layout",
        "block_size",
        "pooled_shape",
        "feature_order",
        "feature_dim",
        "pixel_divisor",
    ],
)
def test_should_refuse_every_missing_recipe_field_instead_of_backfilling_defaults(tmp_path, field):
    raw = teacher()
    del raw["observation_recipe"][field]
    with pytest.raises(ValueError):
        cli._read_teacher(write(tmp_path / "missing.json", raw), 17, visual_observation=True)


@pytest.mark.parametrize(
    "bad",
    [
        "missing",
        "null",
        "list",
        "extra",
        "float_integer",
        "integer_float",
        "list_element_type",
        "equal_bool",
        "list_length",
        "string_array",
        "different_value",
        "nonfinite",
        "order",
    ],
)
def test_should_refuse_null_extra_drifted_or_equal_wrong_type_recipe_json(tmp_path, bad):
    raw = teacher()
    selected = raw["observation_recipe"]
    if bad == "missing":
        del raw["observation_recipe"]
    elif bad == "null":
        raw["observation_recipe"] = None
    elif bad == "list":
        raw["observation_recipe"] = []
    elif bad == "extra":
        selected["unreviewed_option"] = None
    elif bad == "float_integer":
        selected["block_size"] = 8.0
    elif bad == "integer_float":
        selected["pixel_divisor"] = 255
    elif bad == "list_element_type":
        selected["frame_shape"][0] = 128.0
    elif bad == "equal_bool":
        selected["camera_eye"][1] = False
    elif bad == "list_length":
        selected["pooled_shape"].append(1)
    elif bad == "string_array":
        selected["camera_target"] = "-0.1,0.0,0.1"
    elif bad == "different_value":
        selected["camera_near"] = 0.02
    elif bad == "nonfinite":
        selected["camera_fov"] = float("nan")
    else:
        selected["feature_order"].reverse()
    with pytest.raises(ValueError):
        cli._read_teacher(write(tmp_path / "invalid-recipe.json", raw), 17, visual_observation=True)


@pytest.mark.parametrize("duplicate", ["recipe", "qualification"])
def test_should_refuse_duplicate_visual_fields_before_overwriting_ambiguity(tmp_path, duplicate):
    encoded = json.dumps(teacher(), separators=(",", ":"))
    if duplicate == "recipe":
        encoded = encoded.replace('"block_size":8', '"block_size":8,"block_size":8', 1)
    else:
        encoded = encoded.replace('"qualified":true', '"qualified":false,"qualified":true', 1)
    path = tmp_path / "ambiguous.json"
    path.write_text(encoded, encoding="utf8")
    with pytest.raises(ValueError):
        cli._read_teacher(path, 17, visual_observation=True)


@pytest.mark.parametrize(
    "bad",
    [
        "a_context_absent",
        "context_wrong",
        "episode_context",
        "held_role",
        "seed",
        "state_dimension",
        "units",
        "bounds",
        "seam",
        "reward",
        "control",
        "end",
    ],
)
def test_should_refuse_misbound_or_corrupt_visual_receipts(tmp_path, bad):
    raw = teacher()
    episode = raw["candidate_episode"]
    if bad == "a_context_absent":
        del raw["context"]
    elif bad == "context_wrong":
        raw["context"] = "pick-cube-train-19"
    elif bad == "episode_context":
        episode["context"] = "other"
    elif bad == "held_role":
        episode["role"] = "probe_train"
    elif bad == "seed":
        raw["seed"] = 23
    elif bad == "state_dimension":
        episode["spec"]["observation_dim"] = 42
    elif bad == "units":
        episode["spec"]["action_units"] = ["m"] * 4
    elif bad == "bounds":
        episode["spec"]["upper_bounds"] = [2.0] * 4
    elif bad == "seam":
        episode["transitions"][0]["next_observation"][0] = 100.0
    elif bad == "reward":
        episode["transitions"][0]["reward"] = 0.5
    elif bad == "control":
        episode["transitions"][0]["action"][0] = -1.00000001
    else:
        episode["transitions"][-1]["terminated"] = False
    with pytest.raises(ValueError):
        cli._read_teacher(write(tmp_path / "corrupt.json", raw), 17, visual_observation=True)


@pytest.mark.parametrize("mode", ["visual_as_state", "state_as_visual"])
def test_should_refuse_state_rgb_mode_mixing(tmp_path, mode):
    raw = teacher(visual=mode == "visual_as_state")
    if mode == "state_as_visual":
        raw["observation_recipe"] = recipe()
    with pytest.raises(ValueError):
        cli._read_teacher(
            write(tmp_path / "mixed.json", raw), 17, visual_observation=mode == "state_as_visual"
        )


def test_should_preserve_legacy_state_a_context_fallback_with_null_visual_metadata(tmp_path):
    raw = teacher(visual=False)
    del raw["context"]
    result = cli._read_teacher(write(tmp_path / "state.json", raw), 17)
    assert result.spec.observation_dim == 42 and result.context == "pick-cube"


@pytest.mark.parametrize("where", ["initial_low", "joined_high", "final_next_high"])
def test_should_refuse_impossible_rgb_feature_range_without_incidental_seam_errors(tmp_path, where):
    raw = teacher()
    transitions = raw["candidate_episode"]["transitions"]
    if where == "initial_low":
        transitions[0]["observation"][21] = -0.01
    elif where == "joined_high":
        transitions[0]["next_observation"][-1] = 1.01
        transitions[1]["observation"][-1] = 1.01
    else:
        transitions[-1]["next_observation"][21] = 1.01
    with pytest.raises(ValueError, match="RGB.*range"):
        cli._read_teacher(write(tmp_path / "rgb-range.json", raw), 17, visual_observation=True)


def test_should_keep_state_coordinates_unrestricted_by_rgb_suffix_rules(tmp_path):
    raw = teacher(visual=False)
    transitions = raw["candidate_episode"]["transitions"]
    transitions[0]["observation"][21] = -20.0
    transitions[0]["next_observation"][21] = transitions[1]["observation"][21] = 2000.0
    transitions[-1]["next_observation"][21] = -3000.0
    loaded = cli._read_teacher(write(tmp_path / "physical-state.json", raw), 17)
    assert loaded.transitions[0].observation[21] == -20.0
    assert loaded.transitions[1].observation[21] == 2000.0
    assert loaded.transitions[-1].next_observation[21] == -3000.0


def test_should_keep_visual_proprioception_and_goal_in_physical_units(tmp_path):
    raw = teacher()
    transitions = raw["candidate_episode"]["transitions"]
    transitions[0]["observation"][0], transitions[0]["observation"][20] = -123.0, 999.0
    transitions[-1]["next_observation"][20] = 1234.0
    loaded = cli._read_teacher(
        write(tmp_path / "public-prefix.json", raw), 17, visual_observation=True
    )
    assert loaded.transitions[0].observation[0] == -123.0
    assert loaded.transitions[0].observation[20] == 999.0
    assert loaded.transitions[-1].next_observation[20] == 1234.0


@pytest.mark.parametrize(
    "scenario", ["stream", "promotion-rejection", "coupled-neutral", "matched-baselines"]
)
def test_should_refuse_visual_flag_for_old_scenarios_before_io_or_any_runner(monkeypatch, scenario):
    def forbidden(*_args, **_kwargs):
        pytest.fail("invalid visual combination reached IO or a runner")

    for name in (
        "_read_teacher",
        "run_continuous_demo",
        "run_demo",
        "run_rejection_demo",
        "run_coupled_demo",
        "run_matched_demo",
    ):
        monkeypatch.setattr(cli, name, forbidden)
    monkeypatch.setattr(sys, "argv", ["stream", "--scenario", scenario, "--visual-observation"])
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2


@pytest.mark.parametrize("extra", [["--seed", "23"], ["--diagnose-units"], ["--missing-files"]])
def test_should_refuse_visual_continuous_option_conflicts_before_io(monkeypatch, extra):
    def forbidden(*_args, **_kwargs):
        pytest.fail("invalid visual options reached IO or runtime")

    monkeypatch.setattr(cli, "_read_teacher", forbidden)
    monkeypatch.setattr(cli, "run_continuous_demo", forbidden)
    arguments = ["stream", "--scenario", "continuous-nearest", "--visual-observation"]
    if extra != ["--missing-files"]:
        arguments += ["--warm-teacher", "warm.json", "--new-teacher", "new.json", *extra]
    monkeypatch.setattr(sys, "argv", arguments)
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2


def test_should_refuse_mixed_cli_sources_before_fitting_or_creating_runtime(monkeypatch, tmp_path):
    warm = write(tmp_path / "warm.json", teacher())
    new = write(tmp_path / "new.json", teacher(19, visual=False))
    before = warm.read_bytes(), new.read_bytes()
    monkeypatch.setattr(
        demo,
        "fit_training_standardization",
        lambda *_: pytest.fail("mixed source modes reached fitting"),
    )
    monkeypatch.setattr(
        demo,
        "ActorShadowRuntime",
        lambda *_args, **_kwargs: pytest.fail("mixed source modes reached runtime"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stream",
            "--scenario",
            "continuous-nearest",
            "--visual-observation",
            "--warm-teacher",
            str(warm),
            "--new-teacher",
            str(new),
        ],
    )
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2 and (warm.read_bytes(), new.read_bytes()) == before


def test_should_forward_true_to_both_readers_and_keep_default_state_calls_unchanged(
    monkeypatch, tmp_path, capsys
):
    calls = []
    objects = [object(), object()]

    def read(path, seed, **options):
        calls.append((path, seed, options))
        return objects[0 if seed == 17 else 1]

    def run(first, second):
        assert (first, second) == tuple(objects)
        return dict(complete=True)

    monkeypatch.setattr(cli, "_read_teacher", read)
    monkeypatch.setattr(cli, "run_continuous_demo", run)
    paths = [tmp_path / "warm.json", tmp_path / "new.json"]
    base = [
        "stream",
        "--scenario",
        "continuous-nearest",
        "--warm-teacher",
        str(paths[0]),
        "--new-teacher",
        str(paths[1]),
    ]
    for visual in (False, True):
        calls.clear()
        monkeypatch.setattr(sys, "argv", base + (["--visual-observation"] if visual else []))
        cli.main()
        report = json.loads(capsys.readouterr().out)
        assert calls == [
            (paths[0], 17, {"visual_observation": True} if visual else {}),
            (paths[1], 19, {"visual_observation": True} if visual else {}),
        ]
        if visual:
            assert report["input_observation"] == dict(
                mode="public_rgb", recipe=recipe(), per_file_byte_limit=4 * 1024**2
            )
        else:
            assert "input_observation" not in report


def test_should_run_actual_delayed_789d_stream_with_one_charge_frozen_actor_and_correct_costs(
    monkeypatch, tmp_path, capsys
):
    paths = [write(tmp_path / "warm.json", teacher()), write(tmp_path / "new.json", teacher(19))]
    before = [path.read_bytes() for path in paths]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stream",
            "--scenario",
            "continuous-nearest",
            "--visual-observation",
            "--warm-teacher",
            str(paths[0]),
            "--new-teacher",
            str(paths[1]),
        ],
    )
    cli.main()
    captured = capsys.readouterr()
    assert captured.err == ""
    report = json.loads(captured.out)
    assert report["complete"] is True and all(report["checks"].values())
    assert report["input_observation"] == dict(
        mode="public_rgb", recipe=recipe(), per_file_byte_limit=4194304
    )
    assert report["policy"]["spec"]["observation_dim"] == 789
    assert report["policy"]["observation_standardization"]["fitted_rows"] == 2
    assert report["experiment_state"]["completed_updates"] == 1
    assert [event["candidate_prototype_rows"] for event in report["events"]] == [2, 2, 4]
    assert [event["budget_updates"] for event in report["events"]] == [0, 0, 1]
    assert [len(event["applied"]) for event in report["events"]] == [0, 0, 1]
    assert all(
        event["actor_predictions"] == [[-0.5, 0.0, 0.0, -1.0]] * 2 for event in report["events"]
    )
    assert report["readouts"]["candidate_warm"] == [[-1.0, 0.0, 0.0, 1.0], [-0.5, 0.0, 0.0, -1.0]]
    assert report["readouts"]["candidate_new"] == [[1.0, 0.0, 0.0, 1.0], [0.5, 0.0, 0.0, -1.0]]
    work = report["work"]
    expected = dict(
        fit_attempts=1,
        fit_returns=1,
        fit_rows=2,
        fit_feature_values=1578,
        fit_numeric_bytes=12624,
        warm_append_rows=2,
        runtime_append_returns=1,
        runtime_append_rows=2,
        source_rows_ingested=4,
        replayed_training_rows=0,
        optimizer_updates=0,
        actor_prediction_rows=8,
        actor_prediction_bytes=256,
        actor_distance_candidates=16,
        actor_distance_feature_values=12624,
        candidate_prediction_rows=4,
        candidate_prediction_bytes=128,
        candidate_distance_candidates=16,
        candidate_distance_feature_values=12624,
        normalization_calls=8,
        normalization_rows=16,
    )
    assert {key: work[key] for key in expected} == expected
    memory = report["memory"]
    assert memory["source_prototype_bytes"] == memory["actor_prototype_bytes"] == 12688
    assert memory["candidate_prototype_bytes"] == memory["readout_prototype_bytes"] == 25376
    assert memory["prototype_growth_bytes"] == memory["inbox_feature_target_bytes"] == 12688
    assert memory["statistics_bytes_per_owner"] == 12624
    assert (
        report["settings"]["simulator_calls"]
        == report["settings"]["optimizer_updates"]
        == report["settings"]["promotion_calls"]
        == 0
    )
    assert [path.read_bytes() for path in paths] == before
