"""Saved mass input admission and CLI dispatch only; every learner is stubbed."""

from copy import deepcopy
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import sys
from typing import cast

import pytest

from scripts import run_native_stream_demo as cli
from src.core.pick_cube_rgb_features import pick_cube_rgb_recipe
from src.infra.maniskill_continuous import ManiSkillVisualWork
from test_mass_context_interfaces import qualified_result
from test_visual_continuous_stream import teacher as default_teacher


MISSING = object()


def receipt(*, visual=False):
    result = qualified_result()
    if visual:
        episode = result.candidate_episode
        rows = tuple(
            replace(
                row,
                observation=row.observation[:21] + (0.25,) * 768,
                next_observation=row.next_observation[:21] + (0.25,) * 768,
            )
            for row in episode.transitions
        )
        result = replace(
            result,
            candidate_episode=replace(
                episode, spec=replace(episode.spec, observation_dim=789), transitions=rows
            ),
            observation_recipe=pick_cube_rgb_recipe(),
            visual_work=ManiSkillVisualWork(
                3, 3, 3 * 49152, 3 * 789, 3 * 49152, 3 * 789, 3 * 6312, 49152, 0, 0
            ),
        )
    raw = dict(
        asdict(result),
        qualified=True,
        role="train",
        seed=17,
        task="PickCube-v1",
        controller="pd_ee_delta_pos",
        optimizer_updates=0,
        context=result.candidate_episode.context,
    )
    return json.loads(json.dumps(raw))


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


@pytest.fixture(autouse=True)
def forbid_runners(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("input validation or routing executed a learner runner")

    for name in (
        "run_demo",
        "run_coupled_demo",
        "run_matched_demo",
        "run_rejection_demo",
        "run_continuous_demo",
        "run_continuous_learnability_demo",
    ):
        monkeypatch.setattr(cli, name, forbidden)


@pytest.mark.parametrize("visual", [False, True])
def test_should_admit_owned_qualified_mass_episode_and_original_acquisition_metadata(
    tmp_path, visual
):
    raw = receipt(visual=visual)
    path = write(tmp_path / "mass.json", raw)
    before = path.read_bytes()
    episode, metadata = cli._read_mass_teacher(path, visual_observation=visual)
    assert episode.context == raw["context"] and episode.role == "train"
    assert episode.episode_id == raw["candidate_episode"]["episode_id"]
    assert episode.spec.observation_dim == (789 if visual else 42)
    assert episode.transitions[-1].reward == 1.0 and len(episode.transitions) == 2
    expected = dict(
        mode="mass_drift",
        warm_seed=17,
        new_seed=17,
        warm_context="pick-cube",
        new_context=raw["context"],
        mass_context=raw["mass_context"],
        mass_readback=raw["mass_readback"],
        mass_work=raw["mass_work"],
        teacher_work=raw["work"],
        teacher_creation_attempts=1,
        teacher_geometry_attempts=2,
        teacher_geometry_returns=2,
        teacher_actions=2,
        source_file_bytes=len(before),
        per_file_byte_limit=(4 if visual else 1) * 1024**2,
        acquisition_scope="retained_teacher_work_not_new_simulator_work",
    )
    if visual:
        expected["teacher_visual_work"] = raw["visual_work"]
    assert metadata == expected
    metadata["mass_context"]["seed"] = 19
    metadata["mass_readback"]["source_mass_kg"] = 9.0
    if visual:
        metadata["teacher_visual_work"]["current_owned_frame_bytes"] = 0
    object.__setattr__(
        episode.transitions[0], "observation", (999.0,) * episode.spec.observation_dim
    )
    repeated, again = cli._read_mass_teacher(path, visual_observation=visual)
    assert again["mass_context"]["seed"] == 17
    assert again["mass_readback"]["source_mass_kg"] == 0.125
    if visual:
        assert again["teacher_visual_work"]["current_owned_frame_bytes"] == 49152
    assert repeated.transitions[0].observation[0] == 0.0
    assert path.read_bytes() == before


@pytest.mark.parametrize("role", ["inner_guard", "probe_train", "probe_test", "final_test"])
@pytest.mark.parametrize("location", ["top", "episode"])
def test_should_refuse_every_heldout_or_guard_role(tmp_path, role, location):
    raw = receipt()
    raw["role"] = role if location == "top" else "train"
    if location == "episode":
        raw["candidate_episode"]["role"] = role
    with pytest.raises(ValueError):
        cli._read_mass_teacher(write(tmp_path / "role.json", raw))


@pytest.mark.parametrize(
    "path,value",
    [
        (("qualified",), False),
        (("qualified",), 1),
        (("observed_success",), False),
        (("observed_success",), 1),
        (("final_success",), False),
        (("final_success",), 1),
        (("error",), "native failure"),
        (("traceback",), "retained traceback"),
        (("seed",), 19),
        (("seed",), 23),
        (("seed",), 31),
        (("seed",), 37),
        (("seed",), 17.0),
        (("seed",), True),
        (("context",), MISSING),
        (("context",), "pick-cube"),
        (("task",), "Other-v1"),
        (("controller",), "pd_joint_pos"),
        (("candidate_episode", "episode_id"), "pick-cube-task-teacher-17"),
        (("candidate_episode", "context"), "pick-cube"),
        (("guard_seed",), 31),
        (("labels_arrived_at",), 2),
        (("camera_context",), {"context": "pick-cube-camera-y005-train-17", "seed": 17}),
        (("creation_attempts",), True),
        (("creation_attempts",), 2),
        (("geometry_attempts",), 3),
        (("geometry_returns",), 2.0),
        (("teacher_actions",), True),
        (("stages",), ["align"]),
        (("stages",), "align"),
        (("work", "step_attempts"), True),
        (("work", "completed_steps"), 1),
        (("work", "completed_control_physics_steps"), 9),
        (("work", "possible_failed_physics_steps"), 1),
        (("mass_context", "seed"), 17.0),
        (("mass_context", "mass_multiplier"), 2),
        (("mass_context", "semantics"), "density_scaling"),
        (("mass_context", "extra"), 0),
        (("mass_readback", "context", "seed"), True),
        (("mass_readback", "source_mass_kg"), 0),
        (("mass_readback", "source_mass_kg"), 0.0),
        (("mass_readback", "source_mass_kg"), 0.75),
        (("mass_readback", "source_mass_kg"), 0.1),
        (("mass_readback", "applied_mass_kg"), 0.5),
        (("mass_readback", "applied_mass_kg"), 0.25000000000000006),
        (("mass_readback", "extra"), 0),
        (("mass_work", "read_attempts"), True),
        (("mass_work", "read_returns"), 8.0),
        (("mass_work", "read_attempts"), 7),
        (("mass_work", "read_returns"), 7),
        (("mass_work", "write_attempts"), 2),
        (("mass_work", "write_returns"), True),
        (("mass_work", "extra"), 0),
    ],
)
def test_should_refuse_unqualified_misbound_partial_or_untyped_mass_input(tmp_path, path, value):
    raw = receipt()
    change(raw, path, value)
    with pytest.raises(ValueError):
        cli._read_mass_teacher(write(tmp_path / "corrupt.json", raw))


@pytest.mark.parametrize("marker", ["mass_context", "mass_readback", "mass_work"])
@pytest.mark.parametrize("value", [None, MISSING])
def test_should_refuse_missing_original_mass_markers(tmp_path, marker, value):
    raw = receipt()
    change(raw, (marker,), value)
    with pytest.raises(ValueError):
        cli._read_mass_teacher(write(tmp_path / "partial.json", raw))


def test_should_refuse_mathematical_doubling_without_native_float32_readback(tmp_path):
    raw = receipt()
    raw["mass_readback"]["source_mass_kg"] = 0.1
    raw["mass_readback"]["applied_mass_kg"] = 0.2
    with pytest.raises(ValueError):
        cli._read_mass_teacher(write(tmp_path / "unobserved-float32.json", raw))


def test_should_refuse_stripping_all_mass_markers_without_relabeling_the_episode(tmp_path):
    raw = receipt()
    for marker in ("mass_context", "mass_readback", "mass_work"):
        del raw[marker]
    with pytest.raises(ValueError):
        cli._read_mass_teacher(write(tmp_path / "stripped.json", raw))


@pytest.mark.parametrize(
    "path,value",
    [
        (("candidate_episode", "spec", "observation_dim"), 3),
        (("candidate_episode", "spec", "action_units", 0), "other"),
        (("candidate_episode", "spec", "lower_bounds", 0), 0.0),
        (("candidate_episode", "transitions", 0, "step"), True),
        (("candidate_episode", "transitions", 0, "action"), [0.0] * 3),
        (("candidate_episode", "transitions", 0, "action", 0), 2.0),
        (("candidate_episode", "transitions", 0, "reward"), 0.5),
        (("candidate_episode", "transitions", 1, "reward"), 0.0),
        (("candidate_episode", "transitions", 1, "observation", 0), 999.0),
        (("candidate_episode", "transitions", 1, "next_observation", 0), float("inf")),
        (("candidate_episode", "transitions", 0, "terminated"), True),
        (("candidate_episode", "transitions", 1, "terminated"), False),
        (("candidate_episode", "transitions", 1, "truncated"), 0),
    ],
)
def test_should_refuse_corrupt_control_spec_seams_terminal_or_rewards(tmp_path, path, value):
    raw = receipt()
    change(raw, path, value)
    with pytest.raises(ValueError):
        cli._read_mass_teacher(write(tmp_path / "episode.json", raw))


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "extra",
        "wrong_type",
        "wrong_camera",
        "low_rgb",
        "high_terminal_rgb",
        "state_mode",
    ],
)
def test_should_require_exact_base_visual_recipe_and_validate_dropped_terminal_pixels(
    tmp_path, damage
):
    raw = receipt(visual=True)
    if damage == "missing":
        del raw["observation_recipe"]
    elif damage == "extra":
        raw["observation_recipe"]["extra"] = 0
    elif damage == "wrong_type":
        raw["observation_recipe"]["feature_dim"] = 789.0
    elif damage == "wrong_camera":
        raw["observation_recipe"]["camera_eye"][1] = 0.05
    elif damage == "low_rgb":
        for row in raw["candidate_episode"]["transitions"]:
            row["observation"][21] = row["next_observation"][21] = -0.01
    elif damage == "high_terminal_rgb":
        raw["candidate_episode"]["transitions"][-1]["next_observation"][21] = 1.01
    path = write(tmp_path / "visual.json", raw)
    with pytest.raises(ValueError):
        cli._read_mass_teacher(path, visual_observation=damage != "state_mode")


@pytest.mark.parametrize(
    "field",
    [
        "observation_attempts",
        "observation_returns",
        "input_frame_bytes_upper_bound",
        "feature_values_upper_bound",
        "returned_frame_bytes",
        "returned_feature_values",
        "returned_feature_numeric_bytes",
        "current_owned_frame_bytes",
        "frame_export_copies",
        "frame_export_bytes",
    ],
)
def test_should_bind_every_original_visual_work_field_to_teacher_lifecycle(tmp_path, field):
    raw = receipt(visual=True)
    raw["visual_work"][field] += 1
    with pytest.raises(ValueError):
        cli._read_mass_teacher(write(tmp_path / "visual-cost.json", raw), visual_observation=True)


@pytest.mark.parametrize("damage", ["missing", "null", "extra", "float", "bool", "negative"])
def test_should_refuse_missing_untyped_or_unbounded_visual_cost_metadata(tmp_path, damage):
    raw = receipt(visual=True)
    if damage == "missing":
        del raw["visual_work"]
    elif damage == "null":
        raw["visual_work"] = None
    elif damage == "extra":
        raw["visual_work"]["extra"] = 0
    elif damage == "float":
        raw["visual_work"]["observation_returns"] = 3.0
    elif damage == "bool":
        raw["visual_work"]["frame_export_copies"] = False
    else:
        raw["visual_work"]["returned_frame_bytes"] = -1
    with pytest.raises(ValueError):
        cli._read_mass_teacher(write(tmp_path / "visual-types.json", raw), visual_observation=True)


def test_should_refuse_visual_work_attached_to_state_teacher(tmp_path):
    raw = receipt()
    raw["visual_work"] = receipt(visual=True)["visual_work"]
    with pytest.raises(ValueError):
        cli._read_mass_teacher(write(tmp_path / "state-visual-cost.json", raw))


@pytest.mark.parametrize(
    "marker",
    [
        "mass_context",
        "mass_readback",
        "mass_work",
        "guard_seed",
        "labels_arrived_at",
        "camera_context",
    ],
)
def test_should_not_let_default_reader_drop_context_or_guard_markers_after_relabeling(
    tmp_path, marker
):
    raw = default_teacher(17, visual=False)
    raw[marker] = (
        receipt()[marker]
        if marker.startswith("mass_")
        else {"guard_seed": 31, "labels_arrived_at": 2, "camera_context": {"seed": 17}}[marker]
    )
    with pytest.raises(ValueError):
        cli._read_teacher(write(tmp_path / "legacy.json", raw), 17)


def test_should_preserve_default_dataclass_none_fields_without_mass_admission(tmp_path):
    raw = default_teacher(17, visual=False)
    raw.update(mass_context=None, mass_readback=None, mass_work=None)
    episode = cli._read_teacher(write(tmp_path / "default-none.json", raw), 17)
    assert episode.context == "pick-cube" and episode.episode_id == "pick-cube-task-teacher-17"


@pytest.mark.parametrize("visual", [False, True])
@pytest.mark.parametrize("duplicate", ["qualification", "nested_seed"])
def test_should_refuse_duplicate_fields_even_when_later_value_looks_valid(
    tmp_path, visual, duplicate
):
    encoded = json.dumps(receipt(visual=visual))
    if duplicate == "qualification":
        encoded = '{"qualified":false,' + encoded[1:]
    else:
        encoded = encoded.replace('"seed": 17', '"seed": 19, "seed": 17', 1)
    path = tmp_path / "duplicate.json"
    path.write_text(encoded, encoding="utf8")
    with pytest.raises(ValueError, match="duplicate"):
        cli._read_mass_teacher(path, visual_observation=visual)


@pytest.mark.parametrize("visual", [False, True])
def test_should_issue_one_capped_owned_read_and_keep_exact_limit(visual):
    limit = (4 if visual else 1) * 1024**2
    encoded = json.dumps(receipt(visual=visual)).encode("utf8")
    payload = encoded + b" " * (limit - len(encoded))
    reads = []

    class Reader(io.BytesIO):
        def read(self, size=-1):
            reads.append(size)
            return super().read(size)

    class ReceiptPath:
        def open(self, mode):
            assert mode == "rb"
            return Reader(payload)

    _, metadata = cli._read_mass_teacher(cast(Path, ReceiptPath()), visual_observation=visual)
    assert reads == [limit + 1]
    assert metadata["source_file_bytes"] == metadata["per_file_byte_limit"] == limit


@pytest.mark.parametrize("visual", [False, True])
def test_should_refuse_oversized_receipt_before_parsing(tmp_path, monkeypatch, visual):
    limit = (4 if visual else 1) * 1024**2
    path = tmp_path / "large.json"
    path.write_bytes(b" " * (limit + 1))
    monkeypatch.setattr(
        cli.json, "loads", lambda *_args, **_kwargs: pytest.fail("oversize input parsed")
    )
    with pytest.raises(ValueError):
        cli._read_mass_teacher(path, visual_observation=visual)


@pytest.mark.parametrize("bad", [None, 0, 1, "true", []])
def test_should_refuse_untyped_visual_mode_before_file_io(bad):
    class ForbiddenPath:
        def open(self, *_args):
            pytest.fail("invalid visual flag reached IO")

    with pytest.raises(ValueError):
        cli._read_mass_teacher(cast(Path, ForbiddenPath()), visual_observation=bad)


@pytest.mark.parametrize("scenario", ["continuous-nearest", "continuous-mlp-learnability"])
@pytest.mark.parametrize("visual", [False, True])
def test_should_route_fixed_mass17_without_relabeling_and_report_original_acquisition(
    tmp_path, monkeypatch, capsys, scenario, visual
):
    warm_path = write(tmp_path / "warm.json", default_teacher(17, visual=visual))
    raw = receipt(visual=visual)
    new_path = write(tmp_path / "mass.json", raw)
    before = warm_path.read_bytes(), new_path.read_bytes()
    calls = []

    def runner(warm, new, **options):
        calls.append((warm, new, options))
        assert warm.context == "pick-cube" and new.context == raw["context"]
        assert new.episode_id == raw["candidate_episode"]["episode_id"]
        assert warm.spec.observation_dim == new.spec.observation_dim == (789 if visual else 42)
        return dict(complete=True, untouched="stubbed_no_learning")

    target = (
        "run_continuous_demo"
        if scenario == "continuous-nearest"
        else "run_continuous_learnability_demo"
    )
    monkeypatch.setattr(cli, target, runner)
    flags = ["--visual-observation"] if visual else []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stream",
            "--scenario",
            scenario,
            "--warm-teacher",
            str(warm_path),
            "--new-teacher",
            str(new_path),
            "--mass-drift",
            *flags,
        ],
    )
    cli.main()
    report = json.loads(capsys.readouterr().out)
    assert len(calls) == 1 and calls[0][2] == {}
    assert report["untouched"] == "stubbed_no_learning" and report["complete"] is True
    context = report["input_context"]
    assert context["warm_seed"] == context["new_seed"] == 17
    assert context["new_context"] == raw["context"] and context["warm_context"] == "pick-cube"
    for key in ("mass_context", "mass_readback", "mass_work"):
        assert context[key] == raw[key]
    assert context["teacher_work"] == raw["work"]
    assert context["acquisition_scope"] == "retained_teacher_work_not_new_simulator_work"
    assert context["source_file_bytes"] == len(before[1])
    if visual:
        assert report["input_observation"]["mode"] == "public_rgb"
    else:
        assert "input_observation" not in report
    assert (warm_path.read_bytes(), new_path.read_bytes()) == before


@pytest.mark.parametrize("visual", [False, True])
def test_should_compose_existing_phase_binary_and_rgb_options_with_stubbed_mass_mlp(
    tmp_path, monkeypatch, capsys, visual
):
    warm_path = write(tmp_path / "warm.json", default_teacher(17, visual=visual))
    new_path = write(tmp_path / "mass.json", receipt(visual=visual))
    calls = []

    def runner(_warm, _new, **options):
        calls.append(options)
        return dict(complete=True)

    monkeypatch.setattr(cli, "run_continuous_learnability_demo", runner)
    visual_flags = ["--visual-observation", "--rgb-range-normalization"] if visual else []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stream",
            "--scenario",
            "continuous-mlp-learnability",
            "--warm-teacher",
            str(warm_path),
            "--new-teacher",
            str(new_path),
            "--mass-drift",
            "--phase-updates",
            "2",
            "--binary-gripper-axis",
            "3",
            *visual_flags,
        ],
    )
    cli.main()
    assert len(calls) == 1
    expected: dict[str, object] = dict(phase_updates=2, binary_gripper_axis=3)
    if visual:
        expected["rgb_range_recipe"] = pick_cube_rgb_recipe()
    assert calls[0] == expected
    assert json.loads(capsys.readouterr().out)["input_context"]["new_seed"] == 17


@pytest.mark.parametrize("scenario", ["continuous-nearest", "continuous-mlp-learnability"])
@pytest.mark.parametrize("visual", [False, True])
def test_should_preserve_default17_19_routing_and_fieldsets_without_mass_option(
    tmp_path, monkeypatch, capsys, scenario, visual
):
    warm_path = write(tmp_path / "warm.json", default_teacher(17, visual=visual))
    new_path = write(tmp_path / "new.json", default_teacher(19, visual=visual))
    calls = []

    def runner(warm, new, **options):
        calls.append((warm.context, new.context, options))
        return dict(complete=True, preserved=23)

    monkeypatch.setattr(
        cli,
        "_read_mass_teacher",
        lambda *_args, **_kwargs: pytest.fail("default selected mass reader"),
    )
    monkeypatch.setattr(
        cli,
        "run_continuous_demo"
        if scenario == "continuous-nearest"
        else "run_continuous_learnability_demo",
        runner,
    )
    flags = ["--visual-observation"] if visual else []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stream",
            "--scenario",
            scenario,
            "--warm-teacher",
            str(warm_path),
            "--new-teacher",
            str(new_path),
            *flags,
        ],
    )
    cli.main()
    report = json.loads(capsys.readouterr().out)
    assert calls == [("pick-cube", "pick-cube-train-19", {})]
    assert "input_context" not in report and report["preserved"] == 23
    assert not {"mass_context", "mass_work", "mass_readback"} & report.keys()


@pytest.mark.parametrize(
    "flags",
    [
        ["--scenario", "stream"],
        ["--scenario", "promotion-rejection"],
        ["--scenario", "coupled-neutral"],
        ["--scenario", "matched-baselines"],
        ["--scenario", "continuous-nearest", "--seed", "17"],
        ["--scenario", "continuous-nearest", "--diagnose-units"],
        ["--scenario", "continuous-nearest", "--phase-updates", "2"],
        ["--scenario", "continuous-nearest", "--binary-gripper-axis", "3"],
        [
            "--scenario",
            "continuous-mlp-learnability",
            "--visual-observation",
            "--public-vector-only",
        ],
    ],
)
def test_should_refuse_invalid_mass_routing_or_projection_before_reading(monkeypatch, flags):
    def forbidden(*_args, **_kwargs):
        pytest.fail("invalid mass route read teacher input")

    monkeypatch.setattr(cli, "_read_teacher", forbidden)
    monkeypatch.setattr(cli, "_read_mass_teacher", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stream",
            "--warm-teacher",
            "warm.json",
            "--new-teacher",
            "mass.json",
            "--mass-drift",
            *flags,
        ],
    )
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2


def test_should_print_incomplete_runner_receipt_with_acquisition_and_exit_one(
    tmp_path, monkeypatch, capsys
):
    warm_path = write(tmp_path / "warm.json", default_teacher(17, visual=False))
    new_path = write(tmp_path / "mass.json", receipt())
    returned = dict(
        complete=False, error="preserved paid failure", progress=dict(updates_completed=1)
    )
    monkeypatch.setattr(cli, "run_continuous_demo", lambda *_args: deepcopy(returned))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stream",
            "--scenario",
            "continuous-nearest",
            "--warm-teacher",
            str(warm_path),
            "--new-teacher",
            str(new_path),
            "--mass-drift",
        ],
    )
    with pytest.raises(SystemExit) as caught:
        cli.main()
    report = json.loads(capsys.readouterr().out)
    assert caught.value.code == 1
    assert report["complete"] is False and report["error"] == returned["error"]
    assert report["progress"] == returned["progress"]
    assert report["input_context"]["teacher_actions"] == 2


def test_should_reject_invalid_mass_input_before_runner_and_print_no_report(
    tmp_path, monkeypatch, capsys
):
    warm_path = write(tmp_path / "warm.json", default_teacher(17, visual=False))
    raw = receipt()
    raw["mass_readback"]["applied_mass_kg"] = 0.5
    new_path = write(tmp_path / "mass.json", raw)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stream",
            "--scenario",
            "continuous-mlp-learnability",
            "--warm-teacher",
            str(warm_path),
            "--new-teacher",
            str(new_path),
            "--mass-drift",
        ],
    )
    with pytest.raises(SystemExit) as caught:
        cli.main()
    captured = capsys.readouterr()
    assert caught.value.code == 2 and captured.out == ""
