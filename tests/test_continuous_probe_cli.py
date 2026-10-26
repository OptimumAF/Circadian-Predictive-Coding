"""Reserved episode JSON and existing CLI composition; no new probe scheduler.

One actual BP CLI fixture: 11 updates/22 training rows, one fit/two rows/84
values, one initializer/2,944 normal values. Eight probe scopes use 64 prediction
calls/128 rows and eight direction gradients/16 rows. Benchmark conditioning,
residual and actor each use eight calls/16 rows. All other runners are stubbed.
"""

from copy import deepcopy
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import sys

import pytest

from scripts import run_native_stream_demo as cli
from src.infra import continuous_probe_json as reader
from test_continuous_stream_demo import teacher_receipt


def raw_probe(role="probe_train", *, dimension=42, rows=2):
    observations = [[float(index), 0.0] + [0.0] * (dimension - 2) for index in range(rows + 1)]
    if dimension == 1:
        observations = [[float(index)] for index in range(rows + 1)]
    return dict(
        episode_id="reserved-" + role,
        context="reserved-fresh",
        role=role,
        spec=dict(
            observation_dim=dimension,
            action_units=[
                "normalized_root_dx",
                "normalized_root_dy",
                "normalized_root_dz",
                "normalized_gripper",
            ],
            lower_bounds=[-1.0] * 4,
            upper_bounds=[1.0] * 4,
        ),
        transitions=[
            dict(
                step=index,
                observation=list(observations[index]),
                action=[0.25, -0.25, 0.5, -1.0],
                reward=0.0,
                next_observation=list(observations[index + 1]),
                terminated=index == rows - 1,
                truncated=False,
            )
            for index in range(rows)
        ],
    )


def write_probe(path, raw=None, *, role="probe_train"):
    path.write_text(
        json.dumps(raw_probe(role) if raw is None else raw, allow_nan=False), encoding="utf8"
    )
    return path


@pytest.mark.parametrize("role", ["probe_train", "probe_test"])
def test_should_read_one_complete_owned_reserved_episode_and_actual_bytes(tmp_path, role):
    raw = raw_probe(role)
    path = write_probe(tmp_path / "probe.json", raw)
    original = path.read_bytes()
    episode, size = reader.read_continuous_probe_episode(path, role=role)
    assert size == len(original) and path.read_bytes() == original
    assert episode.role == role and episode.episode_id == raw["episode_id"]
    assert episode.context == "reserved-fresh" and episode.spec.observation_dim == 42
    assert episode.transitions[-1].terminated is True
    raw["transitions"][0]["observation"][0] = 100.0
    assert episode.transitions[0].observation[0] == 0.0
    assert type(episode.transitions) is tuple and type(episode.transitions[0].observation) is tuple


@pytest.mark.parametrize("dimension,rows", [(1, 128), (1024, 1)])
def test_should_accept_core_dimension_and_reserved_row_boundaries(tmp_path, dimension, rows):
    path = write_probe(tmp_path / "probe.json", raw_probe(dimension=dimension, rows=rows))
    episode, _ = reader.read_continuous_probe_episode(path, role="probe_train")
    assert len(episode.transitions) == rows and episode.spec.observation_dim == dimension


@pytest.mark.parametrize("role", [None, True, "train", "inner_guard", "final_test", "probe_train "])
def test_should_reject_requested_authority_before_opening_file(monkeypatch, role):
    monkeypatch.setattr(
        Path, "open", lambda *_a, **_k: pytest.fail("invalid requested role opened file")
    )
    with pytest.raises(ValueError):
        reader.read_continuous_probe_episode(Path("unread.json"), role=role)


MISSING_FIELDS = (
    [("episode", name) for name in ("episode_id", "context", "role", "spec", "transitions")]
    + [
        ("spec", name)
        for name in ("observation_dim", "action_units", "lower_bounds", "upper_bounds")
    ]
    + [
        ("transition", name)
        for name in (
            "step",
            "observation",
            "action",
            "reward",
            "next_observation",
            "terminated",
            "truncated",
        )
    ]
)


@pytest.mark.parametrize("level,name", MISSING_FIELDS)
def test_should_refuse_every_missing_json_field_without_default_backfill(tmp_path, level, name):
    raw = raw_probe()
    record = (
        raw if level == "episode" else raw["spec"] if level == "spec" else raw["transitions"][0]
    )
    del record[name]
    path = write_probe(tmp_path / "probe.json", raw)
    with pytest.raises(ValueError):
        reader.read_continuous_probe_episode(path, role="probe_train")


@pytest.mark.parametrize("level", ["episode", "spec", "transition"])
def test_should_refuse_unknown_fields_at_each_object_level(tmp_path, level):
    raw = raw_probe()
    record = (
        raw if level == "episode" else raw["spec"] if level == "spec" else raw["transitions"][0]
    )
    record["qualified"] = True
    with pytest.raises(ValueError):
        reader.read_continuous_probe_episode(
            write_probe(tmp_path / "probe.json", raw), role="probe_train"
        )


@pytest.mark.parametrize(
    "damage",
    [
        "top_list",
        "spec_list",
        "transitions_dict",
        "transition_list",
        "units_text",
        "bounds_text",
        "observation_text",
        "action_dict",
        "step_bool",
        "dimension_float",
        "flag_int",
    ],
)
def test_should_refuse_wrong_json_containers_and_equal_looking_scalar_types(tmp_path, damage):
    raw = raw_probe()
    if damage == "top_list":
        raw = [raw]
    elif damage == "spec_list":
        raw["spec"] = [raw["spec"]]
    elif damage == "transitions_dict":
        raw["transitions"] = {"0": raw["transitions"][0]}
    elif damage == "transition_list":
        raw["transitions"][0] = []
    elif damage == "units_text":
        raw["spec"]["action_units"] = "xxxx"
    elif damage == "bounds_text":
        raw["spec"]["lower_bounds"] = "----"
    elif damage == "observation_text":
        raw["transitions"][0]["observation"] = "0" * 42
    elif damage == "action_dict":
        raw["transitions"][0]["action"] = {"x": 0.25}
    elif damage == "step_bool":
        raw["transitions"][0]["step"] = False
    elif damage == "dimension_float":
        raw["spec"]["observation_dim"] = 42.0
    else:
        raw["transitions"][0]["terminated"] = 0
    with pytest.raises(ValueError):
        reader.read_continuous_probe_episode(
            write_probe(tmp_path / "probe.json", raw), role="probe_train"
        )


@pytest.mark.parametrize(
    "damage",
    [
        "role",
        "empty",
        "too_many",
        "seam",
        "early_terminal",
        "no_terminal",
        "steps",
        "bounds",
        "shape",
        "reversed_bounds",
    ],
)
def test_should_refuse_invalid_episode_integrity_without_relabelling_or_clipping(tmp_path, damage):
    raw = raw_probe(rows=129 if damage == "too_many" else 2)
    if damage == "role":
        raw["role"] = "final_test"
    elif damage == "empty":
        raw["transitions"] = []
    elif damage == "seam":
        raw["transitions"][1]["observation"][0] += 0.1
    elif damage == "early_terminal":
        raw["transitions"][0]["terminated"] = True
    elif damage == "no_terminal":
        raw["transitions"][-1]["terminated"] = False
    elif damage == "steps":
        raw["transitions"][1]["step"] = 2
    elif damage == "bounds":
        raw["transitions"][0]["action"][0] = 1.01
    elif damage == "shape":
        raw["transitions"][0]["action"] = [0.0]
    elif damage == "reversed_bounds":
        raw["spec"]["lower_bounds"][0] = 2.0
    with pytest.raises(ValueError):
        reader.read_continuous_probe_episode(
            write_probe(tmp_path / "probe.json", raw), role="probe_train"
        )


@pytest.mark.parametrize("level", ["episode", "spec", "transition"])
def test_should_refuse_duplicate_fields_before_json_can_overwrite_evidence(tmp_path, level):
    text = json.dumps(raw_probe())
    old, new = {
        "episode": ('"role": "probe_train"', '"role":"probe_test","role":"probe_train"'),
        "spec": ('"observation_dim": 42', '"observation_dim":42,"observation_dim":42'),
        "transition": ('"step": 0', '"step":0,"step":0'),
    }[level]
    path = tmp_path / "probe.json"
    path.write_text(text.replace(old, new, 1), encoding="utf8")
    with pytest.raises(ValueError, match="duplicate"):
        reader.read_continuous_probe_episode(path, role="probe_train")


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "1e400"])
def test_should_refuse_nonfinite_json_constants_and_overflowed_numeric_literals(tmp_path, literal):
    path = tmp_path / "probe.json"
    path.write_text(
        json.dumps(raw_probe()).replace('"reward": 0.0', '"reward":' + literal, 1), encoding="utf8"
    )
    with pytest.raises(ValueError):
        reader.read_continuous_probe_episode(path, role="probe_train")


@pytest.mark.parametrize("payload", [b"\xff", b"{not json}"])
def test_should_refuse_invalid_utf8_and_malformed_json(tmp_path, payload):
    path = tmp_path / "probe.json"
    path.write_bytes(payload)
    with pytest.raises(ValueError):
        reader.read_continuous_probe_episode(path, role="probe_train")


def test_should_take_one_bounded_owned_read_and_report_exact_payload_size(monkeypatch):
    payload = json.dumps(raw_probe()).encode("utf8")
    calls: list[tuple[object, ...]] = []

    class Stream(io.BytesIO):
        def read(self, size=-1):
            calls.append(("read", size))
            return super().read(size)

    def open_file(path, mode):
        calls.append(("open", path, mode))
        return Stream(payload)

    monkeypatch.setattr(Path, "open", open_file)
    path = Path("owned-probe.json")
    episode, size = reader.read_continuous_probe_episode(path, role="probe_train")
    assert episode.episode_id == "reserved-probe_train" and size == len(payload)
    assert calls == [("open", path, "rb"), ("read", reader.MAX_PROBE_FILE_BYTES + 1)]


def test_should_accept_exact_file_cap_and_reject_one_more_byte_before_parsing(
    tmp_path, monkeypatch
):
    payload = json.dumps(raw_probe()).encode("utf8")
    path = tmp_path / "probe.json"
    path.write_bytes(payload + b" " * (reader.MAX_PROBE_FILE_BYTES - len(payload)))
    episode, size = reader.read_continuous_probe_episode(path, role="probe_train")
    assert episode.role == "probe_train" and size == reader.MAX_PROBE_FILE_BYTES == 2 * 1024**2
    path.write_bytes(path.read_bytes() + b" ")
    monkeypatch.setattr(
        reader.json, "loads", lambda *_a, **_k: pytest.fail("oversize input reached parser")
    )
    with pytest.raises(ValueError, match="2MiB"):
        reader.read_continuous_probe_episode(path, role="probe_train")


def arguments(folder, *, probes=True):
    result = [
        "run-native-stream",
        "--scenario",
        "continuous-mlp-learnability",
        "--warm-teacher",
        str(folder / "warm.json"),
        "--new-teacher",
        str(folder / "new.json"),
    ]
    if probes:
        result += [
            "--probe-train",
            str(folder / "ftrain.json"),
            "--probe-test",
            str(folder / "ftest.json"),
        ]
    return result


@pytest.fixture(scope="module")
def actual_cli(tmp_path_factory):
    folder = tmp_path_factory.mktemp("continuous-probe-cli")
    for seed, name in ((17, "warm"), (19, "new")):
        (folder / f"{name}.json").write_text(json.dumps(teacher_receipt(seed)), encoding="utf8")
    write_probe(folder / "ftrain.json")
    write_probe(folder / "ftest.json", role="probe_test")
    original = {path.name: path.read_bytes() for path in folder.iterdir()}
    output = io.StringIO()
    with pytest.MonkeyPatch.context() as patch, redirect_stdout(output):
        patch.setattr(
            sys,
            "argv",
            arguments(folder)
            + ["--phase-updates", "1", "--probe-diagnostics", "--probe-zero-velocity-control"],
        )
        cli.main()
    assert {path.name: path.read_bytes() for path in folder.iterdir()} == original
    return json.loads(output.getvalue()), original


def test_should_execute_real_reader_plan_and_existing_paired_bp_recipe_once(actual_cli):
    report, files = actual_cli
    assert report["complete"] is True and report["progress"]["updates_completed"] == 3
    assert report["work"]["fit_calls"] == 1 and report["work"]["fit_rows"] == 2
    assert report["work"]["optimizer_updates"] == 3 and report["work"]["training_rows"] == 6
    assert [point["benchmark_updates"] for point in report["probes"]] == [0, 1, 2, 3]
    for index, point in enumerate(report["probes"]):
        pair = point["report"]
        assert pair["complete"] is True and pair["source_updates"] == index
        assert pair["initial_scores_exact"] is True
        for key in ("inherited", "zero_velocity"):
            nested = pair[key]
            assert nested["complete"] is True
            assert nested["progress"]["updates_completed"] == 1
            assert len(nested["native_updates"]) == len(nested["applied_updates"]) == 1
            assert nested["applied_updates"][0]["episode_id"] == "reserved-probe_train"
            assert nested["checkpoints"][0]["scores"]["fresh"]["role"] == "probe_test"
            assert nested["checkpoints"][0]["scores"]["fresh_training"]["role"] == "probe_train"
    metadata = report["input_probes"]
    assert metadata["training"] == dict(
        episode_id="reserved-probe_train",
        role="probe_train",
        source_file_bytes=len(files["ftrain.json"]),
    )
    assert metadata["evaluation"] == dict(
        episode_id="reserved-probe_test",
        role="probe_test",
        source_file_bytes=len(files["ftest.json"]),
    )
    assert metadata["per_file_byte_limit"] == reader.MAX_PROBE_FILE_BYTES
    assert (
        "caller_declared" in metadata["reservation"]
        and "no collection qualification" in metadata["reservation"]
    )


@pytest.fixture
def routed(monkeypatch, tmp_path):
    read_teacher = cli._read_teacher
    records = {}
    # Reconstruct core episodes from the real reader, without invoking a model.
    for seed, name in ((17, "warm"), (19, "new")):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(teacher_receipt(seed)), encoding="utf8")
        records[name] = read_teacher(path, seed)
    write_probe(tmp_path / "ftrain.json")
    write_probe(tmp_path / "ftest.json", role="probe_test")
    calls = []

    def run(actual_warm, actual_new, **options):
        calls.append((actual_warm, actual_new, options))
        return dict(complete=True, sentinel="stubbed")

    monkeypatch.setattr(cli, "run_continuous_learnability_demo", run)
    return tmp_path, records, calls


@pytest.mark.parametrize("family", ["bp", "pc", "neutral-cpc", "active-cpc"])
def test_should_forward_reserved_plan_and_existing_family_without_extra_runner_calls(
    routed, monkeypatch, capsys, family
):
    folder, _records, calls = routed
    monkeypatch.setattr(
        sys,
        "argv",
        arguments(folder)
        + [
            "--learner-family",
            family,
            "--phase-updates",
            "2",
            "--probe-updates",
            "3",
            "--probe-diagnostics",
            "--probe-zero-velocity-control",
        ],
    )
    cli.main()
    assert len(calls) == 1
    options = calls[0][2]
    value = options["probe_plan"]
    assert value.updates == 3 and value.diagnostics is True and value.zero_velocity_control is True
    assert value.training.role == "probe_train" and value.evaluation.role == "probe_test"
    assert options["phase_updates"] == 2
    assert options.get("learner_family", "bp") == family
    assert ("learner_family" in options) is (family != "bp")
    assert json.loads(capsys.readouterr().out)["input_probes"]["training"]["role"] == "probe_train"


def test_should_keep_old_default_call_and_report_without_probe_options(routed, monkeypatch, capsys):
    folder, _records, calls = routed
    monkeypatch.setattr(
        cli,
        "read_continuous_probe_episode",
        lambda *_a, **_k: pytest.fail("default CLI opened a probe"),
    )
    monkeypatch.setattr(sys, "argv", arguments(folder, probes=False))
    cli.main()
    assert len(calls) == 1 and calls[0][2] == {}
    assert json.loads(capsys.readouterr().out) == dict(complete=True, sentinel="stubbed")


def test_should_default_only_opted_in_probe_count_to_one(routed, monkeypatch, capsys):
    folder, _records, calls = routed
    monkeypatch.setattr(sys, "argv", arguments(folder))
    cli.main()
    value = calls[0][2]["probe_plan"]
    assert (
        value.updates == 1 and value.diagnostics is False and value.zero_velocity_control is False
    )
    assert len(calls) == 1 and "input_probes" in json.loads(capsys.readouterr().out)


@pytest.mark.parametrize(
    "case",
    [
        "only_train",
        "only_test",
        "count_without_files",
        "diagnostics_without_files",
        "control_without_files",
        "nearest",
        "legacy",
        "zero",
        "over",
        "negative",
        "float",
    ],
)
def test_should_refuse_invalid_probe_cli_options_before_any_file_read(monkeypatch, tmp_path, case):
    argv = arguments(tmp_path, probes=False)
    if case in ("only_train", "only_test"):
        argv += ["--probe-train" if case == "only_train" else "--probe-test", "probe.json"]
    elif case.endswith("without_files"):
        argv += {
            "count_without_files": ["--probe-updates", "1"],
            "diagnostics_without_files": ["--probe-diagnostics"],
            "control_without_files": ["--probe-zero-velocity-control"],
        }[case]
    else:
        argv = arguments(tmp_path)
        if case in ("nearest", "legacy"):
            argv[2] = "continuous-nearest" if case == "nearest" else "stream"
        else:
            argv += [
                "--probe-updates",
                {"zero": "0", "over": "17", "negative": "-1", "float": "1.5"}[case],
            ]
    monkeypatch.setattr(
        cli, "_read_teacher", lambda *_a, **_k: pytest.fail("invalid probe CLI opened teacher")
    )
    monkeypatch.setattr(
        cli,
        "read_continuous_probe_episode",
        lambda *_a, **_k: pytest.fail("invalid probe CLI opened probe"),
    )
    monkeypatch.setattr(
        cli,
        "run_continuous_learnability_demo",
        lambda *_a, **_k: pytest.fail("invalid probe CLI reached runner"),
    )
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2


@pytest.mark.parametrize(
    "damage",
    [
        "train_role",
        "evaluation_role",
        "warm_identity",
        "new_identity",
        "benchmark_context",
        "spec",
        "combined_budget",
    ],
)
def test_should_reject_probe_authority_identity_context_spec_and_budget_before_runner(
    routed, monkeypatch, damage
):
    folder, _records, calls = routed
    training, evaluation = raw_probe(), raw_probe("probe_test")
    if damage == "train_role":
        training["role"] = "final_test"
    elif damage == "evaluation_role":
        evaluation["role"] = "train"
    elif damage == "warm_identity":
        training["episode_id"] = "pick-cube-task-teacher-17"
    elif damage == "new_identity":
        evaluation["episode_id"] = "pick-cube-task-teacher-19"
    elif damage == "benchmark_context":
        training["context"] = evaluation["context"] = "pick-cube"
    elif damage == "spec":
        training, evaluation = raw_probe(dimension=41), raw_probe("probe_test", dimension=41)
    else:
        training, evaluation = (
            raw_probe(dimension=1024, rows=65),
            raw_probe("probe_test", dimension=1024, rows=1),
        )
    write_probe(folder / "ftrain.json", training)
    write_probe(folder / "ftest.json", evaluation)
    monkeypatch.setattr(sys, "argv", arguments(folder))
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2 and calls == []


def test_should_print_incomplete_paid_report_once_and_exit_one_with_input_evidence(
    routed, monkeypatch, capsys
):
    folder, _records, _calls = routed
    calls = []
    partial = dict(
        complete=False,
        error="paid probe stopped",
        progress=dict(updates_completed=1),
        probes=[
            dict(
                report=dict(
                    complete=False,
                    native_updates=[dict(training_rows=2)],
                    progress=dict(updates_completed=1),
                )
            )
        ],
    )

    def run(*_args, **_options):
        calls.append(True)
        return deepcopy(partial)

    monkeypatch.setattr(cli, "run_continuous_learnability_demo", run)
    monkeypatch.setattr(sys, "argv", arguments(folder))
    with pytest.raises(SystemExit) as caught:
        cli.main()
    returned = json.loads(capsys.readouterr().out)
    assert caught.value.code == 1 and calls == [True]
    assert returned["error"] == partial["error"] and returned["probes"] == partial["probes"]
    assert returned["progress"] == dict(updates_completed=1)
    assert returned["input_probes"]["evaluation"]["role"] == "probe_test"


def visual_teachers(monkeypatch, records):
    owned = {}
    for name, episode in records.items():
        owned[name] = replace(
            episode,
            spec=replace(episode.spec, observation_dim=789),
            transitions=tuple(
                replace(
                    row,
                    observation=row.observation + (0.0,) * (789 - 42),
                    next_observation=row.next_observation + (0.0,) * (789 - 42),
                )
                for row in episode.transitions
            ),
        )

    def read(_path, seed, **options):
        assert options == dict(visual_observation=True)
        return owned["warm" if seed == 17 else "new"]

    monkeypatch.setattr(cli, "_read_teacher", read)


@pytest.mark.parametrize("route", ["rgb", "rgb_range", "public21"])
def test_should_bind_visual_or_already_projected_probe_inputs_and_preserve_existing_knobs(
    routed, monkeypatch, capsys, route
):
    folder, records, calls = routed
    visual_teachers(monkeypatch, records)
    dimension = 21 if route == "public21" else 789
    write_probe(folder / "ftrain.json", raw_probe(dimension=dimension))
    write_probe(folder / "ftest.json", raw_probe("probe_test", dimension=dimension))
    argv = arguments(folder) + [
        "--visual-observation",
        "--phase-updates",
        "1",
        "--a-only-control",
        "--binary-gripper-axis",
        "3",
    ]
    if route != "rgb":
        argv.append("--public-vector-only" if route == "public21" else "--rgb-range-normalization")
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    assert len(calls) == 1
    warm, new, options = calls[0]
    value = options["probe_plan"]
    assert (
        warm.spec.observation_dim
        == new.spec.observation_dim
        == value.training.spec.observation_dim
        == dimension
    )
    assert value.training.episode_id == "reserved-probe_train"
    assert value.evaluation.role == "probe_test" and value.training.transitions[
        0
    ].observation == tuple(raw_probe(dimension=dimension)["transitions"][0]["observation"])
    assert (
        options["phase_updates"] == 1
        and options["a_only_control"] is True
        and options["binary_gripper_axis"] == 3
    )
    assert ("rgb_range_recipe" in options) is (route == "rgb_range")
    returned = json.loads(capsys.readouterr().out)
    assert returned["input_observation"]["mode"] == (
        "public_vector21" if route == "public21" else "public_rgb"
    )
    assert "input_probes" in returned
    assert ("input_projection" in returned) is (route == "public21")


@pytest.mark.parametrize("damage", ["joined_low", "terminal_high", "unbounded_prefix"])
def test_should_validate_all_visual_probe_rgb_suffixes_before_runner_without_clipping_prefix(
    routed, monkeypatch, capsys, damage
):
    folder, records, calls = routed
    visual_teachers(monkeypatch, records)
    training, evaluation = raw_probe(dimension=789), raw_probe("probe_test", dimension=789)
    if damage == "joined_low":
        training["transitions"][0]["next_observation"][21] = -0.1
        training["transitions"][1]["observation"][21] = -0.1
    elif damage == "terminal_high":
        evaluation["transitions"][-1]["next_observation"][21] = 1.1
    else:
        for episode in (training, evaluation):
            for row in episode["transitions"]:
                row["observation"][20] = row["next_observation"][20] = -1e6
    write_probe(folder / "ftrain.json", training)
    write_probe(folder / "ftest.json", evaluation)
    monkeypatch.setattr(
        sys, "argv", arguments(folder) + ["--visual-observation", "--rgb-range-normalization"]
    )
    if damage == "unbounded_prefix":
        cli.main()
        assert len(calls) == 1 and json.loads(capsys.readouterr().out)["complete"] is True
        assert calls[0][2]["probe_plan"].training.transitions[0].observation[20] == -1e6
    else:
        with pytest.raises(SystemExit) as caught:
            cli.main()
        assert caught.value.code == 2 and calls == []
        assert "RGB" in capsys.readouterr().err


def test_should_refuse_raw_rgb_probe_when_public_vector_model_requires_already_projected21(
    routed, monkeypatch
):
    folder, records, calls = routed
    visual_teachers(monkeypatch, records)
    write_probe(folder / "ftrain.json", raw_probe(dimension=789))
    write_probe(folder / "ftest.json", raw_probe("probe_test", dimension=789))
    monkeypatch.setattr(
        sys, "argv", arguments(folder) + ["--visual-observation", "--public-vector-only"]
    )
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2 and calls == []
