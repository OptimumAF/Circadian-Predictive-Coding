"""Synthetic receipt intake and CLI routing: no learner, query or SDK work.

All costs inside receipts are explicitly synthetic preserved metadata. Seven real
typed admission calls project347 rows/694 views each (2429/4858 total); every
source stage is a controlled incomplete return, with zero optimizer/model work.
"""

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from scripts import joint_acquisition_inputs as intake
from scripts import joint_acquisition_robot_learning as workflow
from scripts import run_continuous_robot_learning as cli
from src.adapters import numpy_continuous_bc as bp
from src.adapters.continuous_learnability_demo import ContinuousLearnabilityResult
from src.core.curated_readiness_corrections import DEFINITION, PROVENANCE, GEOMETRY_DEFINITION
from src.core.pick_cube_readiness_corrections import GOAL_CONSISTENT_OBSERVED_GRASP_PROVENANCE
from src.core.pick_cube_readiness_teacher import (
    OBSERVED_GRASP_DEFINITION,
    DEFINITION as READINESS_DEFINITION,
)
from src.core.pick_cube_taskspace_history import PickCubeTaskSpaceHistoryRecipe
from src.core.readiness_stage_diagnostics import ReadinessStageBinding
from src.infra import maniskill_continuous as native
from test_joint_readiness_corrections import inputs, episode
from test_joint_train20_rehearsal import retained


KINDS = (
    "warm_teacher",
    "new_teacher",
    "corrections",
    "curated_source",
    "curated_labels",
    "retained_teacher",
)
LIMIT = 1024**2


def typed_inputs():
    warm, corrections, curated = inputs(a_rows=15, b_rows=50, c_rows=50)
    new = episode("b", rows=22)
    new = replace(
        new,
        episode_id="pick-cube-readiness-teacher-19",
        transitions=tuple(replace(row, action=(-0.25, 0.0, 0.0, 1.0)) for row in new.transitions),
    )
    return warm, ReadinessStageBinding(new, ("align",) * 22), corrections, curated, retained(15)


def teacher_receipt(binding, seed, *, observed):
    value = asdict(binding.episode)
    value["transitions"] = [
        dict(row, reward=float(i == len(value["transitions"]) - 1))
        for i, row in enumerate(value["transitions"])
    ]
    count = len(value["transitions"])
    raw = dict(
        candidate_episode=value,
        qualified=True,
        observed_success=True,
        final_success=True,
        seed=seed,
        context=binding.episode.context,
        role="train",
        task="PickCube-v1",
        controller="pd_ee_delta_pos",
        creation_attempts=1,
        geometry_attempts=count,
        geometry_returns=count,
        teacher_actions=count,
        stages=list(binding.stages),
        work=dict(
            step_attempts=count,
            completed_steps=count,
            completed_control_physics_steps=5 * count,
            possible_failed_physics_steps=0,
        ),
        error=None,
        traceback=None,
        observation_recipe=None,
        visual_work=None,
        teacher_definition=OBSERVED_GRASP_DEFINITION if observed else READINESS_DEFINITION,
    )
    if observed:
        raw["observed_grasp"] = [row.observation[18] == 1.0 for row in binding.episode.transitions]
    return raw


def native_receipt(value, seed, updates, training_rows):
    return dict(
        complete=True,
        task="PickCube-v1",
        controller="pd_ee_delta_pos",
        seed=seed,
        context=value.context,
        role="train",
        policy="frozen_continuous_bc",
        episode=asdict(value),
        source_updates=updates,
        source_training_rows=training_rows,
        optimizer_updates=0,
        frozen_state_exact=True,
        original_owner_exact=True,
        action_history_exact=True,
        creation_attempts=1,
        creation_returns=1,
        action_history=dict(
            recipe=asdict(PickCubeTaskSpaceHistoryRecipe()),
            pending=False,
            state=dict(previous_action=value.transitions[-1].action, closing_count=6),
            work=dict(
                encoding_attempts=50,
                encoding_returns=50,
                committed_controls=50,
                source_values_validated=2100,
                model_values_emitted=600,
                model_numeric_bytes_emitted=4800,
            ),
        ),
        explicit_reset=f"seed{seed}_if_construction_reaches_port_reset",
        work=dict(
            step_attempts=50,
            completed_steps=50,
            completed_control_physics_steps=250,
            possible_failed_physics_steps=0,
        ),
        sparse_return=0.0,
        any_success=False,
        final_success=False,
        error=None,
        traceback=None,
    )


@pytest.fixture(scope="module")
def raw_receipts():
    warm, new, corrections, curated, extra = typed_inputs()
    b = native_receipt(corrections.episode, 19, 16, 296)
    b["corrections"] = dict(
        provenance=GOAL_CONSISTENT_OBSERVED_GRASP_PROVENANCE,
        train_seed=19,
        teacher_definition=DEFINITION,
        geometry_attempts=50,
        geometry_returns=50,
        teacher_action_attempts=50,
        teacher_actions=50,
        teacher_decision_steps=50,
        labels=[
            dict(asdict(label), observed_at=2 * i, labels_arrived_at=2 * i + 1)
            for i, label in enumerate(corrections.labels)
        ],
        observed_grasp=[False] * 50,
        batch=dict(train_seed=19, episode_id="pick-cube-smoke-19", rows=50, labels_arrived_at=99),
    )
    c = native_receipt(curated.episode, 17, 24, 816)
    export = dict(
        complete=True,
        source_bytes_unchanged=True,
        source_sha256="bound when written",
        source_path="historical-metadata-only-do-not-open.json",
        early_release_refused=True,
        curation=dict(
            complete=True,
            source_episode_id="pick-cube-smoke-17",
            source_context="pick-cube",
            source_role="train",
            source_available_at=100,
            started_at=100,
            source_observed_at=[None] * 50,
            teacher_definition=DEFINITION,
            provenance=PROVENANCE,
            geometry_definition=GEOMETRY_DEFINITION,
            teacher_queries_attempted=50,
            teacher_queries_returned=50,
            work=dict(
                geometry_attempts=50,
                geometry_returns=50,
                source_values_validated=2100,
                teacher_state_steps=50,
            ),
            labels=[asdict(label) for label in curated.labels],
        ),
        work=dict(
            new_teacher_initializations=1,
            teacher_queries_attempted=50,
            teacher_queries_returned=50,
            geometry_rows=50,
            geometry_source_values=2100,
            relative_geometry_values=450,
            relative_geometry_numeric_bytes=3600,
            target_values=200,
            target_numeric_bytes=1600,
            model_projection_rows=50,
            projection_raw_views=100,
            projection_raw_values=4200,
            projection_model_values=1200,
            projection_model_numeric_bytes=9600,
            history_advances=50,
        ),
        stage_coverage=dict(align=50, descend=0, grasp=0, lift=0, settle=0),
        target_differs_from_command_rows=50,
        grasp_positive_source_rows=0,
    )
    return (
        teacher_receipt(warm, 17, observed=True),
        teacher_receipt(new, 19, observed=False),
        b,
        c,
        export,
        teacher_receipt(extra, 20, observed=True),
    )


def encode(value):
    return json.dumps(value, allow_nan=False, sort_keys=True).encode("utf8")


@pytest.fixture
def receipts(tmp_path, raw_receipts):
    raw = deepcopy(raw_receipts)
    raw[4]["source_sha256"] = hashlib.sha256(encode(raw[3])).hexdigest()
    payloads = tuple(encode(value) for value in raw)
    paths = tuple(tmp_path / (kind + ".json") for kind in KINDS)
    for path, payload in zip(paths, payloads, strict=True):
        path.write_bytes(payload)
    return paths, payloads, raw


def forbidden(*args, **kwargs):
    pytest.fail("receipt intake reached a learner, update, native port or unrelated reader")


@pytest.fixture(autouse=True)
def no_learning(monkeypatch):
    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "__init__", forbidden)
    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "train_batch", forbidden)
    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "train_head_gauss_newton_batch", forbidden)
    monkeypatch.setattr(workflow, "run_joint_readiness_stream", forbidden)
    monkeypatch.setattr(workflow, "run_joint_readiness_corrections", forbidden)
    monkeypatch.setattr(workflow, "run_frozen_rollout", forbidden)
    monkeypatch.setattr(native, "_make_environment", forbidden)


@pytest.fixture(scope="module", autouse=True)
def admission_work():
    counts = dict(own_calls=0, retained_calls=0, projection_rows=0, observation_projections=0)
    own = workflow._own_inputs
    prepare = workflow.prepare_joint_readiness_data

    def owned(*args, **options):
        result = own(*args, **options)
        counts["own_calls"] += 1
        counts["projection_rows"] += 217
        counts["observation_projections"] += 434
        return result

    def prepared(*args, **options):
        result = prepare(*args, **options)
        counts["retained_calls"] += 1
        counts["projection_rows"] += len(result.rows)
        counts["observation_projections"] += result.work["observation_projections"]
        return result

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(workflow, "_own_inputs", owned)
        patch.setattr(workflow, "prepare_joint_readiness_data", prepared)
        yield counts
    if os.environ.get("JAC_FIXTURE_LEDGER"):
        with Path(os.environ["JAC_FIXTURE_LEDGER"]).open("x", encoding="utf8") as output:
            json.dump(counts, output, sort_keys=True)
            output.write("\n")
    assert counts["own_calls"] == counts["retained_calls"] == 7
    assert counts["projection_rows"] == 2429 and counts["observation_projections"] == 4858


def test_should_read_exactly_six_capped_byte_payloads_and_preserve_qualified_roles_clocks_costs(
    receipts, monkeypatch
):
    paths, payloads, _ = receipts
    opened: list[Path] = []
    reads: list[tuple[Path, int]] = []
    original = Path.open

    class Handle:
        def __init__(self, actual, path):
            self.actual, self.path = actual, path

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.actual.close()

        def read(self, size):
            reads.append((self.path, size))
            return self.actual.read(size)

    def open_path(path, mode="r", *args, **options):
        actual = original(path, mode, *args, **options)
        if mode == "rb":
            opened.append(path)
            return Handle(actual, path)
        return actual

    monkeypatch.setattr(Path, "open", open_path)
    values, metadata = intake.read_joint_acquisition_inputs(*paths)
    assert opened == list(paths) and reads == [(path, LIMIT + 1) for path in paths]
    assert tuple(len(value.episode.transitions) for value in values) == (15, 22, 50, 50, 15)
    assert all(value.episode.role == "train" for value in values)
    assert values[-1].episode.context == "pick-cube-train-20"
    assert (
        metadata["schema_version"] == 1
        and metadata["protocol_id"] == "joint_acquisition_receipts_v1"
    )
    assert metadata["files"] == [
        dict(
            kind=kind,
            path=str(path),
            sha256=hashlib.sha256(payload).hexdigest(),
            source_file_bytes=len(payload),
            per_file_byte_limit=LIMIT,
        )
        for kind, path, payload in zip(KINDS, paths, payloads, strict=True)
    ]
    assert values[2].labels[-1].labels_arrived_at == 99
    assert values[3].labels[-1].queried_at == 198 and values[3].labels[-1].arrived_at == 199
    assert all(label.source_observed_at is None for label in values[3].labels)
    assert values[2].labels[0].target != values[2].episode.transitions[0].action
    assert values[3].labels[0].target != values[3].episode.transitions[0].action
    assert metadata["corrections"]["teacher_actions"] == 50
    assert metadata["curation"]["work"]["teacher_queries_returned"] == 50
    assert "labels" not in metadata["corrections"] and "episode" not in metadata["curated_source"]
    json.dumps(metadata, allow_nan=False)


def test_should_return_detached_receipt_metadata_and_domain_inputs(receipts):
    paths, payloads, _ = receipts
    values, metadata = intake.read_joint_acquisition_inputs(*paths)
    metadata["teachers"]["warm"]["teacher_work"]["completed_steps"] = 999
    object.__setattr__(values[0].episode.transitions[0], "reward", 0.5)
    again, retained_metadata = intake.read_joint_acquisition_inputs(*paths)
    assert again[0].episode.transitions[0].reward == 0.0
    assert retained_metadata["teachers"]["warm"]["teacher_work"]["completed_steps"] == 15
    assert tuple(path.read_bytes() for path in paths) == payloads


@pytest.mark.parametrize(
    "fault", ["duplicate_top", "duplicate_nested", "nan", "overflow", "utf8", "oversize", "list"]
)
def test_should_refuse_ambiguous_nonfinite_or_oversized_receipt_bytes(receipts, fault):
    paths, _, raw = receipts
    encoded = encode(raw[0])
    if fault == "duplicate_top":
        encoded = b'{"qualified":false,' + encoded[1:]
    elif fault == "duplicate_nested":
        encoded = encoded.replace(
            b'"completed_steps": 15', b'"completed_steps": 14,"completed_steps": 15'
        )
    elif fault in ("nan", "overflow"):
        encoded = encoded.replace(
            b'"reward": 0.0', b'"reward": ' + (b"NaN" if fault == "nan" else b"1e999"), 1
        )
    elif fault == "utf8":
        encoded = b"\xff"
    elif fault == "oversize":
        encoded = b" " * (LIMIT + 1)
    else:
        encoded = b"[]"
    paths[0].write_bytes(encoded)
    with pytest.raises(ValueError):
        intake.read_joint_acquisition_inputs(*paths)


@pytest.mark.parametrize(
    "fault",
    [
        "warm_role",
        "retained_role",
        "unqualified",
        "request_error",
        "extra_episode_field",
        "integer_normalization",
        "native_role",
        "correction_clock",
        "correction_work",
        "provenance",
        "command",
        "source_hash",
        "curated_clock",
        "invented_clock",
        "curated_work",
        "source_counts",
        "curated_role",
    ],
)
def test_should_refuse_relabelled_unarrived_or_corrupted_original_evidence(receipts, fault):
    paths, _, raw = receipts
    selected = 0
    if fault == "warm_role":
        raw[0]["candidate_episode"]["role"] = "probe_test"
    elif fault == "retained_role":
        selected = 5
        raw[5]["role"] = "final_test"
    elif fault == "unqualified":
        raw[0]["qualified"] = False
    elif fault == "request_error":
        raw[0]["train_request_error"] = "collector returned another TRAIN binding"
    elif fault == "extra_episode_field":
        raw[0]["candidate_episode"]["new_authority"] = True
    elif fault == "integer_normalization":
        values = list(raw[0]["candidate_episode"]["transitions"][0]["observation"])
        values[0] = 0
        raw[0]["candidate_episode"]["transitions"][0]["observation"] = values
    elif fault == "native_role":
        selected = 2
        raw[2]["episode"]["role"] = "inner_guard"
    elif fault in ("correction_clock", "correction_work", "provenance", "command"):
        selected = 2
        if fault == "correction_clock":
            raw[2]["corrections"]["labels"][-1]["labels_arrived_at"] = 100
        elif fault == "correction_work":
            raw[2]["corrections"]["teacher_actions"] = 49
        elif fault == "provenance":
            raw[2]["corrections"]["provenance"] = "qualified_demonstration"
        else:
            raw[2]["episode"]["transitions"][-1]["action"] = [0.1, 0.0, 0.0, -1.0]
    elif fault == "source_counts":
        selected = 3
        raw[3]["source_updates"] = 32
    else:
        selected = 4
        if fault == "source_hash":
            raw[4]["source_sha256"] = "0" * 64
        elif fault == "curated_clock":
            raw[4]["curation"]["labels"][-1]["arrived_at"] = 200
        elif fault == "invented_clock":
            raw[4]["curation"]["labels"][0]["source_observed_at"] = 0
        elif fault == "curated_work":
            raw[4]["work"]["teacher_queries_returned"] = 49
        else:
            raw[4]["curation"]["source_role"] = "probe_train"
    paths[selected].write_bytes(encode(raw[selected]))
    with pytest.raises(ValueError):
        intake.read_joint_acquisition_inputs(*paths)


def cli_args(paths, output):
    return [
        "--joint-acquisition",
        "--output",
        str(output),
        "--warm-teacher",
        str(paths[0]),
        "--new-teacher",
        str(paths[1]),
        "--joint-corrections",
        str(paths[2]),
        "--curated-source",
        str(paths[3]),
        "--curated-labels",
        str(paths[4]),
        "--retained-teacher",
        str(paths[5]),
    ]


@pytest.mark.parametrize("complete,explicit_defaults", [(True, False), (False, True)])
def test_should_route_six_actual_receipts_once_and_print_complete_or_paid_partial_result(
    receipts, monkeypatch, tmp_path, capsys, complete, explicit_defaults
):
    paths, _, _ = receipts
    reads: list[tuple[Path, ...]] = []
    runs: list[tuple[Path, tuple[Any, ...], dict[str, Any]]] = []
    read = cli.read_joint_acquisition_inputs
    output = tmp_path / "cli.json"

    def read_inputs(*args):
        reads.append(args)
        return read(*args)

    def run(path, *values, **options):
        runs.append((path, values, options))
        assert tuple(len(value.episode.transitions) for value in values) == (15, 22, 50, 50, 15)
        assert set(options) == {"input_receipts"}
        report = dict(
            complete=complete,
            input_receipts=options["input_receipts"],
            progress=dict(updates_completed=0),
            fixture="routing only; zero learner work",
        )
        with path.open("x", encoding="utf8") as handle:
            json.dump(report, handle, allow_nan=False)
        return report

    monkeypatch.setattr(cli, "read_joint_acquisition_inputs", read_inputs)
    monkeypatch.setattr(cli, "run_joint_acquisition_rollouts", run)
    flags = ["--learner-family", "bp", "--phase-updates", "8"] if explicit_defaults else []
    code = cli.main(cli_args(paths, output) + flags)
    assert code == (0 if complete else 1)
    assert reads == [paths] and len(runs) == 1
    printed = json.loads(capsys.readouterr().out)
    assert printed == json.loads(output.read_text(encoding="utf8"))
    assert [item["path"] for item in printed["input_receipts"]["files"]] == [
        str(path) for path in paths
    ]


def paid_failure(*args, **options):
    return ContinuousLearnabilityResult(
        dict(
            complete=False,
            error="controlled paid-prefix fixture",
            fixture="no actual optimizer work",
            progress=dict(updates_completed=1),
            paid_prefix=[dict(index=0)],
        ),
        None,
    )


def test_should_run_actual_cli_reader_and_typed_admission_then_save_paid_failure_metadata(
    receipts, monkeypatch, tmp_path, capsys
):
    paths, _, _ = receipts
    output = tmp_path / "typed-cli.json"
    monkeypatch.setattr(workflow, "run_joint_readiness_stream", paid_failure)
    assert cli.main(cli_args(paths, output)) == 1
    report = json.loads(capsys.readouterr().out)
    assert report == json.loads(output.read_text(encoding="utf8"))
    assert report["source"]["paid_prefix"] == [dict(index=0)]
    assert report["source"]["progress"]["updates_completed"] == 1
    assert report["native_attempts"] == report["rollouts"] == []
    assert report["input_receipts"]["protocol_id"] == "joint_acquisition_receipts_v1"
    assert report["input_admission_work"]["projection_rows"] == 347


@pytest.mark.parametrize(
    "flags",
    [
        ["--learner-family", "pc"],
        ["--phase-updates", "1"],
        ["--visual-observation"],
        ["--binary-gripper"],
        ["--observation-scale-floors"] + ["0.1"] * 42,
        ["--canonical-state-quaternions"],
        ["--action-history", "executed"],
        ["--lighting-drift"],
        ["--a-only-control"],
        ["--readiness-teachers"],
        ["--taskspace-history", "executed"],
        ["--observed-grasp-warm-teacher"],
        ["--readiness-stage-residuals"],
        ["--stage-balanced-packets"],
    ],
)
def test_should_refuse_every_legacy_mode_before_any_joint_or_teacher_read(
    monkeypatch, tmp_path, flags
):
    monkeypatch.setattr(cli, "read_joint_acquisition_inputs", forbidden)
    monkeypatch.setattr(cli, "_read_teacher", forbidden)
    monkeypatch.setattr(cli, "_read_readiness_teacher", forbidden)
    paths = tuple(tmp_path / (kind + ".json") for kind in KINDS)
    output = tmp_path / "absent.json"
    with pytest.raises(SystemExit) as raised:
        cli.main(cli_args(paths, output) + flags)
    assert raised.value.code == 2 and not output.exists()


@pytest.mark.parametrize(
    "flag", ["--joint-corrections", "--curated-source", "--curated-labels", "--retained-teacher"]
)
def test_should_require_all_joint_paths_before_reading(monkeypatch, tmp_path, flag):
    monkeypatch.setattr(cli, "read_joint_acquisition_inputs", forbidden)
    paths = tuple(tmp_path / (kind + ".json") for kind in KINDS)
    args = cli_args(paths, tmp_path / "missing.json")
    index = args.index(flag)
    del args[index : index + 2]
    with pytest.raises(SystemExit) as raised:
        cli.main(args)
    assert raised.value.code == 2


@pytest.mark.parametrize(
    "flag", ["--joint-corrections", "--curated-source", "--curated-labels", "--retained-teacher"]
)
def test_should_refuse_joint_only_paths_without_selected_mode_before_legacy_reads(
    monkeypatch, tmp_path, flag
):
    monkeypatch.setattr(cli, "read_joint_acquisition_inputs", forbidden)
    monkeypatch.setattr(cli, "_read_teacher", forbidden)
    with pytest.raises(SystemExit) as raised:
        cli.main(
            [
                "--output",
                str(tmp_path / "absent.json"),
                "--warm-teacher",
                "missing17",
                "--new-teacher",
                "missing19",
                flag,
                "joint-only.json",
            ]
        )
    assert raised.value.code == 2


def test_should_report_corrupt_receipt_as_cli_admission_error_without_creating_output(
    receipts, monkeypatch, tmp_path
):
    paths, _, raw = receipts
    raw[5]["qualified"] = False
    paths[5].write_bytes(encode(raw[5]))
    monkeypatch.setattr(cli, "run_joint_acquisition_rollouts", forbidden)
    output = tmp_path / "no-output.json"
    with pytest.raises(SystemExit) as raised:
        cli.main(cli_args(paths, output))
    assert raised.value.code == 2 and not output.exists()


@pytest.mark.parametrize("fault", ["list", "string", "nan", "foreign_value", "oversize"])
def test_should_refuse_invalid_optional_metadata_before_preflight_or_output(
    monkeypatch, tmp_path, fault
):
    metadata: Any = {
        "list": [],
        "string": "not an object",
        "nan": {"value": float("nan")},
        "foreign_value": {"value": object()},
        "oversize": {"value": "x" * (256 * 1024)},
    }[fault]
    monkeypatch.setattr(workflow, "_own_inputs", forbidden)
    output = tmp_path / "invalid-meta.json"
    with pytest.raises((ValueError, TypeError)):
        workflow.run_joint_acquisition_rollouts(output, *typed_inputs(), input_receipts=metadata)
    assert not output.exists()


@pytest.mark.parametrize("mode", ["omitted", "none", "metadata"])
def test_should_preserve_default_schema_and_save_owned_metadata_before_paid_failure(
    receipts, monkeypatch, tmp_path, mode
):
    paths, _, _ = receipts
    values, metadata = intake.read_joint_acquisition_inputs(*paths)
    original = deepcopy(metadata)
    options: dict[str, Any] = (
        {} if mode == "omitted" else dict(input_receipts=None if mode == "none" else metadata)
    )
    monkeypatch.setattr(workflow, "run_joint_readiness_stream", paid_failure)
    output = tmp_path / "metadata.json"
    report = workflow.run_joint_acquisition_rollouts(output, *values, **options)
    assert report["complete"] is False and report["source"]["paid_prefix"] == [dict(index=0)]
    assert report["unreturned_source_work_unknown"] is False
    assert json.loads(output.read_text(encoding="utf8"))["source"] == report["source"]
    if mode == "metadata":
        assert report["input_receipts"] == metadata and report["input_receipts"] is not metadata
        report["input_receipts"]["files"][0]["sha256"] = "edited returned copy"
        assert metadata == original
    else:
        assert "input_receipts" not in report


@pytest.mark.parametrize("fault", ["original_content", "owned_content", "owned_identity"])
def test_should_guard_original_and_owned_receipts_after_a_returned_paid_failure(
    receipts, monkeypatch, tmp_path, fault
):
    paths, _, _ = receipts
    values, metadata = intake.read_joint_acquisition_inputs(*paths)
    captured: list[dict[str, Any]] = []
    write = workflow._write

    def observed_write(path, report, **options):
        captured.append(report)
        return write(path, report, **options)

    def failing_source(*args, **options):
        if fault == "original_content":
            metadata["files"][0]["sha256"] = "changed original receipt"
        elif fault == "owned_content":
            captured[0]["input_receipts"]["files"][0]["sha256"] = "changed owned receipt"
        else:
            captured[0]["input_receipts"] = deepcopy(captured[0]["input_receipts"])
        return paid_failure()

    monkeypatch.setattr(workflow, "_write", observed_write)
    monkeypatch.setattr(workflow, "run_joint_readiness_stream", failing_source)
    output = tmp_path / "changed-metadata.json"
    report = workflow.run_joint_acquisition_rollouts(output, *values, input_receipts=metadata)
    assert report["complete"] is False and "state_error" in report
    assert report["source"]["paid_prefix"] == [dict(index=0)]
    assert report["source"]["error"] == "controlled paid-prefix fixture"
    assert report["native_attempts"] == report["rollouts"] == []
    assert report["acquisition_attempted"] is False
