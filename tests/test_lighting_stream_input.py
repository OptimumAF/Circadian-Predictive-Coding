"""Lighting receipt admission and CLI dispatch; every learner runner is stubbed.

Per module invocation: 107 bounded payload reads (each <=4MiB+1, including two
state reads <=1MiB+1), six stub-runner calls, and 12 small fixture byte-identity
reads. Zero model constructors, fitting, forwards, SGD, model RNG or native IO.
"""

from copy import deepcopy
from dataclasses import asdict
import io
import json
from pathlib import Path
import sys
from typing import Any, cast

import pytest

from scripts import run_native_stream_demo as cli
from src.core.pick_cube_rgb_features import pick_cube_rgb_recipe
from test_lighting_context_interfaces import qualified_result
from test_mass_stream_input import receipt as mass_receipt
from test_visual_continuous_stream import teacher as default_teacher


MISSING = object()
LIMIT = 4 * 1024**2


def receipt(*, optimizer_field=True):
    result = qualified_result()
    raw = dict(
        asdict(result),
        qualified=True,
        role="train",
        seed=17,
        task="PickCube-v1",
        controller="pd_ee_delta_pos",
        context=result.candidate_episode.context,
    )
    if optimizer_field:
        raw["optimizer_updates"] = 0
    return json.loads(json.dumps(raw))


def write(path, raw):
    path.write_text(json.dumps(raw), encoding="utf8")
    return path


def change(raw, path, value):
    owner = raw
    for name in path[:-1]:
        owner = owner[name]
    if value is MISSING:
        del owner[path[-1]]
    else:
        owner[path[-1]] = value


@pytest.fixture(autouse=True)
def forbid_runners(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("receipt validation or routing executed a learner runner")

    for name in (
        "run_demo",
        "run_coupled_demo",
        "run_matched_demo",
        "run_rejection_demo",
        "run_continuous_demo",
        "run_continuous_learnability_demo",
    ):
        monkeypatch.setattr(cli, name, forbidden)


@pytest.mark.parametrize("optimizer_field", [False, True])
def test_should_release_owned_lighting_train_pairs_and_original_paid_metadata(
    tmp_path, optimizer_field
):
    raw = receipt(optimizer_field=optimizer_field)
    path = write(tmp_path / "lighting.json", raw)
    before = path.read_bytes()
    episode, metadata = cli._read_lighting_teacher(path, visual_observation=True)
    assert episode.spec.observation_dim == 789 and episode.role == "train"
    assert (
        episode.episode_id == raw["candidate_episode"]["episode_id"]
        and episode.context == raw["context"]
    )
    assert len(episode.transitions) == 2 and episode.transitions[-1].reward == 1.0
    expected = dict(
        mode="lighting_drift",
        warm_seed=17,
        new_seed=17,
        warm_context="pick-cube",
        new_context=raw["context"],
        lighting_context=raw["lighting_context"],
        lighting_readback=raw["lighting_readback"],
        lighting_work=raw["lighting_work"],
        teacher_work=raw["work"],
        teacher_visual_work=raw["visual_work"],
        teacher_creation_attempts=1,
        teacher_geometry_attempts=2,
        teacher_geometry_returns=2,
        teacher_actions=2,
        source_file_bytes=len(before),
        per_file_byte_limit=LIMIT,
        acquisition_scope="retained_teacher_work_not_new_simulator_work",
    )
    assert json.loads(json.dumps(metadata)) == expected
    metadata["lighting_context"]["seed"] = 19
    assert metadata["lighting_readback"]["context"]["seed"] == 17
    metadata["lighting_readback"]["source_ambient_rgb"] = (9.0,) * 3
    metadata["teacher_visual_work"]["returned_feature_values"] = 0
    object.__setattr__(episode.transitions[0], "observation", (999.0,) * 789)
    repeated, again = cli._read_lighting_teacher(path, visual_observation=True)
    assert again["lighting_context"]["seed"] == again["lighting_readback"]["context"]["seed"] == 17
    assert tuple(again["lighting_readback"]["source_ambient_rgb"]) == (1.0, 0.5, 0.25)
    assert again["teacher_visual_work"]["returned_feature_values"] == 2367
    assert repeated.transitions[0].observation != episode.transitions[0].observation
    assert path.read_bytes() == before


@pytest.mark.parametrize("role", ["inner_guard", "probe_train", "probe_test", "final_test"])
@pytest.mark.parametrize("location", ["top", "episode"])
def test_should_refuse_heldout_and_guard_receipts_at_both_qualification_layers(
    tmp_path, role, location
):
    raw = receipt()
    raw["role"] = role if location == "top" else "train"
    if location == "episode":
        raw["candidate_episode"]["role"] = role
    with pytest.raises(ValueError):
        cli._read_lighting_teacher(write(tmp_path / "role.json", raw), visual_observation=True)


@pytest.mark.parametrize(
    "path,value",
    [
        (("qualified",), False),
        (("qualified",), 1),
        (("observed_success",), False),
        (("final_success",), False),
        (("error",), "native failure"),
        (("traceback",), "paid failure"),
        (("seed",), 19),
        (("seed",), 17.0),
        (("context",), "pick-cube"),
        (("candidate_episode", "episode_id"), "pick-cube-task-teacher-17"),
        (("guard_seed",), 31),
        (("camera_context",), {"seed": 17}),
        (("mass_context",), {"seed": 17}),
        (("creation_attempts",), True),
        (("geometry_returns",), 2.0),
        (("teacher_actions",), 1),
        (("stages",), ["descend", "align"]),
        (("stages",), ["align", 1]),
        (("lighting_context", "ambient_multiplier"), 0.50000001),
        (("lighting_readback", "extra"), 0),
        (("lighting_readback", "source_ambient_rgb", 0), 1),
        (("lighting_readback", "applied_ambient_rgb", 0), 0.25),
        (("lighting_work", "read_returns"), 7),
        (("lighting_work", "refresh_returns"), True),
        (("lighting_work", "discarded_reset_observations"), 0),
        (("optimizer_updates",), True),
        (("controller",), "pd_joint_pos"),
    ],
)
def test_should_refuse_unqualified_misbound_or_untyped_original_lighting_receipts(
    tmp_path, path, value
):
    raw = receipt()
    change(raw, path, value)
    with pytest.raises(ValueError):
        cli._read_lighting_teacher(write(tmp_path / "corrupt.json", raw), visual_observation=True)


@pytest.mark.parametrize("marker", ["lighting_context", "lighting_readback", "lighting_work"])
@pytest.mark.parametrize("value", [None, MISSING])
def test_should_refuse_each_missing_original_lighting_binding(tmp_path, marker, value):
    raw = receipt()
    change(raw, (marker,), value)
    with pytest.raises(ValueError):
        cli._read_lighting_teacher(write(tmp_path / "partial.json", raw), visual_observation=True)


@pytest.mark.parametrize("location", ["declaration", "readback"])
@pytest.mark.parametrize("field", [None, "context", "seed", "ambient_multiplier", "semantics"])
def test_should_refuse_context_defaults_backfilling_missing_recorded_fields(
    tmp_path, location, field
):
    raw = receipt()
    context = (
        raw["lighting_context"]
        if location == "declaration"
        else raw["lighting_readback"]["context"]
    )
    if field is None:
        context.clear()
    else:
        del context[field]
    with pytest.raises(ValueError):
        cli._read_lighting_teacher(
            write(tmp_path / "lost-context-field.json", raw), visual_observation=True
        )


def test_should_refuse_float64_mathematical_half_without_native_float32_measurement(tmp_path):
    raw = receipt()
    raw["lighting_readback"]["source_ambient_rgb"][0] = 0.1
    raw["lighting_readback"]["applied_ambient_rgb"][0] = 0.05
    with pytest.raises(ValueError):
        cli._read_lighting_teacher(
            write(tmp_path / "unobserved.json", raw), visual_observation=True
        )


@pytest.mark.parametrize(
    "path,value",
    [
        (("candidate_episode", "spec", "observation_dim"), 42),
        (("candidate_episode", "transitions", 0, "action", 0), 2.0),
        (("candidate_episode", "transitions", 0, "terminated"), True),
        (("candidate_episode", "transitions", 1, "observation", 0), 999.0),
        (("candidate_episode", "transitions", 1, "reward"), 0.0),
    ],
)
def test_should_refuse_corrupted_control_dimensions_seams_or_complete_success(
    tmp_path, path, value
):
    raw = receipt()
    change(raw, path, value)
    with pytest.raises(ValueError):
        cli._read_lighting_teacher(write(tmp_path / "episode.json", raw), visual_observation=True)


@pytest.mark.parametrize("damage", ["missing", "extra", "different_camera", "terminal_rgb"])
def test_should_require_complete_base_recipe_and_validate_terminal_rgb(tmp_path, damage):
    raw = receipt()
    if damage == "missing":
        del raw["observation_recipe"]
    elif damage == "extra":
        raw["observation_recipe"]["extra"] = 0
    elif damage == "different_camera":
        raw["observation_recipe"]["camera_eye"][1] = 0.05
    else:
        raw["candidate_episode"]["transitions"][-1]["next_observation"][21] = 1.01
    with pytest.raises(ValueError):
        cli._read_lighting_teacher(write(tmp_path / "recipe.json", raw), visual_observation=True)


@pytest.mark.parametrize("damage", ["missing", "bytes", "export_bool", "extra"])
def test_should_bind_typed_visual_costs_to_complete_teacher_acquisition(tmp_path, damage):
    raw = receipt()
    if damage == "missing":
        del raw["visual_work"]
    elif damage == "bytes":
        raw["visual_work"]["returned_frame_bytes"] += 1
    elif damage == "export_bool":
        raw["visual_work"]["frame_export_copies"] = False
    else:
        raw["visual_work"]["extra"] = 0
    with pytest.raises(ValueError):
        cli._read_lighting_teacher(
            write(tmp_path / "visual-cost.json", raw), visual_observation=True
        )


@pytest.mark.parametrize("duplicate", ["qualification", "nested_seed"])
def test_should_refuse_ambiguous_duplicate_json_even_when_last_value_looks_valid(
    tmp_path, duplicate
):
    encoded = json.dumps(receipt())
    encoded = (
        '{"qualified":false,' + encoded[1:]
        if duplicate == "qualification"
        else encoded.replace('"seed": 17', '"seed": 19, "seed": 17', 1)
    )
    path = tmp_path / "duplicate.json"
    path.write_text(encoded, encoding="utf8")
    with pytest.raises(ValueError, match="duplicate"):
        cli._read_lighting_teacher(path, visual_observation=True)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_should_refuse_nonfinite_json_even_in_unrelated_retained_metadata(tmp_path, value):
    raw = receipt()
    raw["unrelated_readout"] = value
    with pytest.raises(ValueError):
        cli._read_lighting_teacher(write(tmp_path / "nonfinite.json", raw), visual_observation=True)


def test_should_use_one_capped_owned_read_and_accept_exact_four_mib_boundary():
    payload = json.dumps(receipt()).encode("utf8")
    payload += b" " * (LIMIT - len(payload))
    reads, opens = [], []

    class Reader(io.BytesIO):
        def read(self, size=-1):
            reads.append(size)
            return super().read(size)

    class ReceiptPath:
        def open(self, mode):
            opens.append(mode)
            return Reader(payload)

    _, metadata = cli._read_lighting_teacher(cast(Path, ReceiptPath()), visual_observation=True)
    assert opens == ["rb"] and reads == [LIMIT + 1]
    assert metadata["source_file_bytes"] == metadata["per_file_byte_limit"] == LIMIT


def test_should_refuse_one_byte_oversize_before_json_parsing(tmp_path, monkeypatch):
    path = tmp_path / "oversize.json"
    path.write_bytes(b" " * (LIMIT + 1))
    monkeypatch.setattr(
        cli.json, "loads", lambda *_args, **_kwargs: pytest.fail("oversize payload parsed")
    )
    with pytest.raises(ValueError):
        cli._read_lighting_teacher(path, visual_observation=True)


@pytest.mark.parametrize("visual", [False, None, 1, "true", []])
def test_should_require_explicit_native_true_visual_flag_before_any_io(visual: Any):
    class ForbiddenPath:
        def open(self, *_args):
            pytest.fail("nonvisual lighting input reached file IO")

    with pytest.raises(ValueError):
        cli._read_lighting_teacher(cast(Path, ForbiddenPath()), visual_observation=visual)


@pytest.mark.parametrize("reader", ["default", "mass"])
@pytest.mark.parametrize("marker", ["lighting_context", "lighting_readback", "lighting_work"])
def test_should_not_drop_lighting_markers_on_otherwise_valid_default_or_mass_routes(
    tmp_path, reader, marker
):
    raw = default_teacher(17) if reader == "default" else mass_receipt(visual=True)
    raw[marker] = receipt()[marker]
    path = write(tmp_path / "misbound.json", raw)
    with pytest.raises(ValueError):
        if reader == "default":
            cli._read_teacher(path, 17, visual_observation=True)
        else:
            cli._read_mass_teacher(path, visual_observation=True)


def test_should_preserve_appended_none_lighting_fields_on_default_and_mass_inputs(tmp_path):
    for name, raw in (("default", default_teacher(17)), ("mass", mass_receipt(visual=True))):
        raw.update(lighting_context=None, lighting_readback=None, lighting_work=None)
        path = write(tmp_path / f"{name}.json", raw)
        episode = (
            cli._read_teacher(path, 17, visual_observation=True)
            if name == "default"
            else cli._read_mass_teacher(path, visual_observation=True)[0]
        )
        assert episode.role == "train" and len(episode.transitions) == 2


@pytest.mark.parametrize("scenario", ["continuous-nearest", "continuous-mlp-learnability"])
def test_should_route_fixed_lighting17_and_report_original_acquisition_without_learning(
    tmp_path, monkeypatch, capsys, scenario
):
    warm = write(tmp_path / "warm.json", default_teacher(17))
    raw = receipt(optimizer_field=False)
    new = write(tmp_path / "lighting.json", raw)
    before = warm.read_bytes(), new.read_bytes()
    calls = []

    def runner(old, fresh, **options):
        calls.append((old, fresh, options))
        return dict(complete=True, preserved="stubbed_no_learning")

    monkeypatch.setattr(
        cli,
        "run_continuous_demo"
        if scenario == "continuous-nearest"
        else "run_continuous_learnability_demo",
        runner,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stream",
            "--scenario",
            scenario,
            "--warm-teacher",
            str(warm),
            "--new-teacher",
            str(new),
            "--visual-observation",
            "--lighting-drift",
        ],
    )
    cli.main()
    report = json.loads(capsys.readouterr().out)
    assert len(calls) == 1 and calls[0][2] == {}
    assert (calls[0][0].context, calls[0][1].context) == ("pick-cube", raw["context"])
    assert calls[0][0].spec.observation_dim == calls[0][1].spec.observation_dim == 789
    assert report["complete"] is True and report["preserved"] == "stubbed_no_learning"
    context = report["input_context"]
    assert context["warm_seed"] == context["new_seed"] == 17 and context["mode"] == "lighting_drift"
    for key in ("lighting_context", "lighting_readback", "lighting_work"):
        assert context[key] == raw[key]
    assert (
        context["teacher_visual_work"] == raw["visual_work"]
        and context["teacher_work"] == raw["work"]
    )
    assert context["source_file_bytes"] == len(before[1])
    assert context["acquisition_scope"] == "retained_teacher_work_not_new_simulator_work"
    assert report["input_observation"]["mode"] == "public_rgb"
    assert (warm.read_bytes(), new.read_bytes()) == before


def test_should_compose_existing_a_only_phase_binary_and_rgb_range_options(
    tmp_path, monkeypatch, capsys
):
    warm = write(tmp_path / "warm.json", default_teacher(17))
    new = write(tmp_path / "lighting.json", receipt())
    calls: list[dict[str, Any]] = []

    def runner(*_args, **options):
        calls.append(options)
        return dict(complete=True)

    monkeypatch.setattr(cli, "run_continuous_learnability_demo", runner)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stream",
            "--scenario",
            "continuous-mlp-learnability",
            "--warm-teacher",
            str(warm),
            "--new-teacher",
            str(new),
            "--visual-observation",
            "--lighting-drift",
            "--a-only-control",
            "--phase-updates",
            "2",
            "--binary-gripper-axis",
            "3",
            "--rgb-range-normalization",
        ],
    )
    cli.main()
    assert calls == [
        dict(
            a_only_control=True,
            phase_updates=2,
            binary_gripper_axis=3,
            rgb_range_recipe=pick_cube_rgb_recipe(),
        )
    ]
    assert json.loads(capsys.readouterr().out)["input_context"]["new_seed"] == 17


def test_should_preserve_default17_19_dispatch_and_report_without_lighting_option(
    tmp_path, monkeypatch, capsys
):
    warm = write(tmp_path / "warm.json", default_teacher(17))
    new = write(tmp_path / "new.json", default_teacher(19))
    calls = []
    monkeypatch.setattr(
        cli,
        "_read_lighting_teacher",
        lambda *_args, **_kwargs: pytest.fail("default route used lighting reader"),
    )

    def runner(a, b):
        calls.append((a.context, b.context))
        return dict(complete=True, unchanged=23)

    monkeypatch.setattr(cli, "run_continuous_demo", runner)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stream",
            "--scenario",
            "continuous-nearest",
            "--warm-teacher",
            str(warm),
            "--new-teacher",
            str(new),
            "--visual-observation",
        ],
    )
    cli.main()
    report = json.loads(capsys.readouterr().out)
    assert calls == [("pick-cube", "pick-cube-train-19")]
    assert report["unchanged"] == 23 and "input_context" not in report


def test_should_preserve_existing_state_mass_route_with_no_lighting_fields(
    tmp_path, monkeypatch, capsys
):
    warm = write(tmp_path / "warm.json", default_teacher(17, visual=False))
    new = write(tmp_path / "mass.json", mass_receipt())
    calls = []
    monkeypatch.setattr(
        cli,
        "_read_lighting_teacher",
        lambda *_args, **_kwargs: pytest.fail("mass route used lighting reader"),
    )

    def runner(a, b):
        calls.append((a.context, b.context))
        return dict(complete=True)

    monkeypatch.setattr(cli, "run_continuous_demo", runner)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stream",
            "--scenario",
            "continuous-nearest",
            "--warm-teacher",
            str(warm),
            "--new-teacher",
            str(new),
            "--mass-drift",
        ],
    )
    cli.main()
    report = json.loads(capsys.readouterr().out)
    assert len(calls) == 1 and report["input_context"]["mode"] == "mass_drift"
    assert not any(key.startswith("lighting_") for key in report["input_context"])


@pytest.mark.parametrize(
    "flags",
    [
        ["--scenario", "continuous-nearest"],
        ["--scenario", "stream", "--visual-observation"],
        ["--scenario", "continuous-nearest", "--visual-observation", "--mass-drift"],
        [
            "--scenario",
            "continuous-mlp-learnability",
            "--visual-observation",
            "--public-vector-only",
        ],
        ["--scenario", "continuous-nearest", "--visual-observation", "--seed", "17"],
        ["--scenario", "continuous-nearest", "--visual-observation", "--phase-updates", "2"],
        ["--scenario", "continuous-nearest", "--visual-observation", "--a-only-control"],
        ["--scenario", "continuous-nearest", "--visual-observation", "--diagnose-units"],
    ],
)
def test_should_refuse_invalid_lighting_cli_allocation_before_reading_any_payload(
    monkeypatch, flags
):
    def forbidden(*_args, **_kwargs):
        pytest.fail("invalid lighting route read teacher payload")

    for name in ("_read_teacher", "_read_mass_teacher", "_read_lighting_teacher"):
        monkeypatch.setattr(cli, name, forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stream",
            "--warm-teacher",
            "warm.json",
            "--new-teacher",
            "lighting.json",
            "--lighting-drift",
            *flags,
        ],
    )
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2


def test_should_print_incomplete_runner_receipt_and_original_acquisition_then_exit_one(
    tmp_path, monkeypatch, capsys
):
    warm = write(tmp_path / "warm.json", default_teacher(17))
    new = write(tmp_path / "lighting.json", receipt())
    returned = dict(
        complete=False, error="retained paid update failure", progress=dict(updates_completed=1)
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
            str(warm),
            "--new-teacher",
            str(new),
            "--visual-observation",
            "--lighting-drift",
        ],
    )
    with pytest.raises(SystemExit) as caught:
        cli.main()
    report = json.loads(capsys.readouterr().out)
    assert caught.value.code == 1 and report["complete"] is False
    assert report["error"] == returned["error"] and report["progress"] == returned["progress"]
    assert report["input_context"]["teacher_actions"] == 2


def test_should_refuse_missing_lighting_teacher_path_before_any_payload_read(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("missing path allocation reached receipt IO")

    for name in ("_read_teacher", "_read_mass_teacher", "_read_lighting_teacher"):
        monkeypatch.setattr(cli, name, forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stream",
            "--scenario",
            "continuous-nearest",
            "--warm-teacher",
            "warm.json",
            "--visual-observation",
            "--lighting-drift",
        ],
    )
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2


def test_should_reject_unqualified_lighting_before_runner_and_print_no_report(
    tmp_path, monkeypatch, capsys
):
    warm = write(tmp_path / "warm.json", default_teacher(17))
    raw = receipt()
    raw["lighting_readback"]["applied_ambient_rgb"][0] = 0.25
    new = write(tmp_path / "lighting.json", raw)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stream",
            "--scenario",
            "continuous-mlp-learnability",
            "--warm-teacher",
            str(warm),
            "--new-teacher",
            str(new),
            "--visual-observation",
            "--lighting-drift",
        ],
    )
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2 and capsys.readouterr().out == ""


@pytest.mark.parametrize("location", ["top", "readback"])
@pytest.mark.parametrize("missing", [None, "context", "seed", "mass_multiplier", "semantics"])
def test_should_refuse_mass_context_default_backfill_in_both_original_declarations(
    tmp_path, location, missing
):
    raw = mass_receipt(visual=True)
    parent = raw if location == "top" else raw["mass_readback"]
    key = "mass_context" if location == "top" else "context"
    if missing is None:
        parent[key] = {}
    else:
        del parent[key][missing]
    with pytest.raises(ValueError, match="complete original typed fields"):
        cli._read_mass_teacher(
            write(tmp_path / "missing-mass-context.json", raw), visual_observation=True
        )
