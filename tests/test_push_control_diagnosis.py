"""Saved Push D12 support and exact-input errors use actual command history.

Synthetic TRAIN records contain no learner or simulator. Forecast:four complete
public diagnoses/eight comparisons,390 identity-copy rows/4680 coordinates and
48144 full12 distance/exact coordinates. One overflow refusal attempts the first
18x3 comparison only. All admission faults must stop before numeric description.
"""

from copy import deepcopy
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import pytest

from scripts import diagnose_learned_robot_control as diagnosis
from src.core.continuous_demonstration import ContinuousEpisode, ContinuousTransition
from src.core.cube_taskspace_state import CUBE_TASKSPACE_DEFINITION, cube_taskspace_spec
from src.core.pick_cube_action_history import ActionHistoryState, advance_action_history


@pytest.fixture(scope="module", autouse=True)
def ledger():
    counts = dict(
        comparison_attempts=0,
        comparison_returns=0,
        input_rows_attempted=0,
        input_rows_returned=0,
        distance_coordinates_attempted=0,
        distance_coordinates_returned=0,
    )
    describe = diagnosis._describe_control

    def described(reference, student, *args, **options):
        old, new = len(reference.transitions), len(student.transitions)
        counts["comparison_attempts"] += 1
        counts["input_rows_attempted"] += old + new
        counts["distance_coordinates_attempted"] += old * new * 12
        result = describe(reference, student, *args, **options)
        counts["comparison_returns"] += 1
        counts["input_rows_returned"] += old + new
        counts["distance_coordinates_returned"] += old * new * 12
        return result

    def forbidden(*args, **options):
        pytest.fail("Push diagnosis attempted fitted normalization")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(diagnosis, "_describe_control", described)
        patch.setattr(diagnosis, "standardize_observations", forbidden)
        yield counts
    if os.environ.get("PDG_FIXTURE_LEDGER"):
        with Path(os.environ["PDG_FIXTURE_LEDGER"]).open("x", encoding="utf8") as output:
            json.dump(counts, output, sort_keys=True)
            output.write("\n")
    assert counts == dict(
        comparison_attempts=9,
        comparison_returns=8,
        input_rows_attempted=411,
        input_rows_returned=390,
        distance_coordinates_attempted=48792,
        distance_coordinates_returned=48144,
    )


def episode(name, count, *, executed, target=None):
    """Targets may differ from the commands that produced these observations."""
    history = ActionHistoryState()
    transitions = []
    for index in range(count):
        physical = (0.0, 0.2 + index * 0.001, 0.0, -0.8, 0.6, 0.0, 0.0)
        current = physical + (*history.previous_action, history.closing_count / 6)
        history = advance_action_history(history, executed)
        next_physical = (0.0, 0.2 + (index + 1) * 0.001, 0.0, -0.8, 0.6, 0.0, 0.0)
        following = next_physical + (*history.previous_action, history.closing_count / 6)
        transitions.append(
            ContinuousTransition(
                index,
                current,
                executed if target is None else target,
                1.0 if index == count - 1 and name == "push-cube-teacher-17" else 0.0,
                following,
                index == count - 1 and name == "push-cube-teacher-17",
                index == count - 1 and name != "push-cube-teacher-17",
            )
        )
    return ContinuousEpisode(
        name, "push-cube-train-17", "train", cube_taskspace_spec(), tuple(transitions)
    )


def native_work(count):
    return dict(
        construction_attempts=1,
        construction_returns=1,
        reset_attempts=1,
        reset_returns=1,
        step_attempts=count,
        completed_steps=count,
        completed_control_physics_steps=5 * count,
        possible_failed_physics_steps=0,
        state_encoding_attempts=count + 1,
        state_encoding_returns=count + 1,
        grasp_query_attempts=count + 1,
        grasp_query_returns=count + 1,
        source_contact_force_query_attempts_upper_bound=2 * (count + 1),
        source_contact_force_query_returns=2 * (count + 1),
        close_attempts=1,
        close_returns=1,
    )


def records(count=3):
    expert = episode("push-cube-teacher-17", 18, executed=(0.0, 0.0, 0.0, -1.0))
    student = episode("push-frozen-mixed120", count, executed=(0.5, 0.0, 0.0, 1.0))
    packet = episode(
        "push-cube-correction-targets-17",
        50,
        executed=(0.5, 0.0, 0.0, 1.0),
        target=(0.25, 0.0, 0.0, -1.0),
    )
    # Final metadata is kept explicit so each admission refusal has one cause.
    rollout: dict[str, Any] = dict(
        complete=True,
        error=None,
        source_unchanged=True,
        unreturned_prediction_possible=False,
        unreturned_native_work_unknown=False,
        prediction_attempts=count,
        prediction_returns=count,
        role="train",
        seed=17,
        task_id="PushCube-v1",
        prediction_outputs=[
            dict(
                dtype="float64",
                shape=[1, 4],
                values=[str(value) for value in row.action],
                values_omitted=0,
            )
            for row in student.transitions
        ],
        control_attempts=count,
        control_returns=count,
        episode=asdict(student),
        transitions=[asdict(row) for row in student.transitions],
        sparse_return=0.0,
        any_success=False,
        final_success=False,
        metadata=dict(
            task_id="PushCube-v1",
            episode_id=student.episode_id,
            seed=17,
            role="train",
            context=student.context,
            horizon=50,
            control_physics_steps_per_control=5,
            recipe=dict(
                definition=CUBE_TASKSPACE_DEFINITION,
                feature_dim=12,
                displacement_scale=0.1,
                displacement_frame="world",
                history="executed",
                closing_limit=6,
            ),
            controller=dict(
                mode="pd_ee_delta_pos",
                frame="root_translation",
                normalized_delta=True,
                no_target=True,
            ),
            reward=dict(
                source="info.success",
                mode="sparse",
                rule="xy_distance_lt_0.1_and_object_z_lt_0.025",
            ),
        ),
        work=native_work(count),
    )
    study: dict[str, Any] = dict(
        protocol="push_cube_prefix_group_balanced_correction_train_v1",
        complete=True,
        error=None,
        source_unchanged=True,
        unreturned_stage_work_unknown=False,
        training=asdict(expert),
        native_push=rollout,
        correction_continuation=dict(
            complete=True,
            source_exact=True,
            actor_exact=True,
            unreturned_training_work_unknown=False,
            initial_counts=dict(updates=112, training_rows=9832),
            final_counts=dict(updates=120, training_rows=11016),
        ),
        correction_fit_protocol=dict(
            initial_counts=[112, 9832],
            final_counts=[120, 11016],
            rows=148,
            updates=8,
        ),
    )
    return json.loads(json.dumps(study, allow_nan=False)), packet


@pytest.fixture(scope="module")
def completed():
    study, packet = records()
    before = deepcopy(study), asdict(packet)
    result = diagnosis.diagnose_push_control(study, packet)
    return result, study, packet, before


def test_should_describe_both_full12_references_without_mutating_saved_inputs(completed):
    result, study, packet, before = completed
    assert result["complete"] is True
    assert result["definition"] == "saved_train_push_taskspace12_control_gap_v1"
    assert result["source_counts"] == dict(updates=120, training_rows=11016)
    assert tuple(result["comparisons"]) == ("expert", "corrections")
    assert study == before[0] and asdict(packet) == before[1]
    for name, rows in (("expert", 18), ("corrections", 50)):
        value = result["comparisons"][name]
        assert value["definition"] == "saved_train_push_taskspace12_exact_input_fit_and_support_v1"
        assert value["rows"] == dict(teacher=rows, student=3)
        assert value["reset_observation_exact"] is True
        assert tuple(value["observation_blocks"]) == ("physical7", "history5")
        assert "teacher_episode" not in value
        assert value["recorded_outcome"] == dict(
            sparse_return=0.0, any_success=False, final_success=False
        )


def test_should_score_only_exact_input_actions_and_keep_student_history_in_target_packet(completed):
    result, _, packet, _ = completed
    expert = result["comparisons"]["expert"]["exact_observation_matches"]
    corrections = result["comparisons"]["corrections"]["exact_observation_matches"]
    assert [row["student_step"] for row in expert] == [0]
    assert [row["student_step"] for row in corrections] == [0, 1, 2]
    assert expert[0]["action_error"] == [0.5, 0.0, 0.0, 2.0]
    assert expert[0]["desired_translation_delta_error_m"] == [0.05, 0.0, 0.0]
    for row in corrections:
        assert row["labels_consistent"] is True
        assert row["action_error"] == [0.25, 0.0, 0.0, 2.0]
        assert row["desired_translation_delta_error_m"] == [0.025, 0.0, 0.0]
    assert packet.transitions[1].observation[7:11] == (0.5, 0.0, 0.0, 1.0)
    assert packet.transitions[1].action == (0.25, 0.0, 0.0, -1.0)
    assert packet.transitions[1].observation[11] == 0.0


def test_should_measure_all12_support_coordinates_and_separate_history_from_geometry(completed):
    result, study, _, _ = completed
    actual = result["comparisons"]["expert"]
    reference = study["training"]["transitions"]
    student = study["native_push"]["transitions"]
    expected_steps, expected_rms = [], []
    for row in student:
        distances = [
            math.fsum(
                (a - b) ** 2 for a, b in zip(row["observation"], old["observation"], strict=True)
            )
            / 12
            for old in reference
        ]
        selected = min(range(18), key=distances.__getitem__)
        expected_steps.append(selected)
        expected_rms.append(math.sqrt(distances[selected]))
    assert actual["nearest_taskspace12_reference_steps"] == expected_steps
    assert actual["nearest_taskspace12_normalized_rms"] == pytest.approx(expected_rms)
    assert actual["first_nonzero_taskspace12_support_distance"] == 1
    assert actual["observation_blocks"]["physical7"][
        "outside_training_range_coordinates_by_step"
    ] == [0, 0, 0]
    assert actual["observation_blocks"]["history5"][
        "outside_training_range_coordinates_by_step"
    ] == [0, 2, 2]


def test_should_account_identity_copies_distances_exact_comparisons_and_sequential_peak(completed):
    result = completed[0]
    for name, rows in (("expert", 18), ("corrections", 50)):
        work = result["comparisons"][name]["work"]
        assert (
            work["normalization_calls"]
            == work["normalization_rows"]
            == work["normalization_coordinates"]
            == 0
        )
        assert work["identity_copy_calls"] == 2
        assert work["identity_copy_rows"] == rows + 3
        assert work["identity_copy_coordinates"] == (rows + 3) * 12
        assert work["taskspace12_distance_coordinates"] == rows * 3 * 12
        assert work["exact_match_coordinate_comparisons"] == rows * 3 * 12
        assert work["selected_working_payload_bound_bytes"] <= 4 * 1024**2
        for field in (
            "model_initializations",
            "fits",
            "optimizer_updates",
            "forwards",
            "gradients",
            "rng_draws",
            "simulator_controls",
        ):
            assert work[field] == 0
    assert result["work"]["identity_copy_calls"] == 4
    assert result["work"]["identity_copy_rows"] == 74
    assert result["work"]["taskspace12_distance_coordinates"] == 2448
    assert result["work"]["selected_peak_payload_bound_bytes"] == max(
        item["work"]["selected_working_payload_bound_bytes"]
        for item in result["comparisons"].values()
    )


def test_should_report_conflicting_exact_labels_without_selecting_one_as_ground_truth():
    study, packet = records()
    rows = list(packet.transitions)
    shared = rows[1].observation
    rows[1] = replace(rows[1], next_observation=shared)
    rows[2] = replace(rows[2], observation=shared, action=(-0.25, 0.0, 0.0, -1.0))
    packet = replace(packet, transitions=tuple(rows))
    result = diagnosis.diagnose_push_control(study, packet)
    known = result["comparisons"]["corrections"]["exact_observation_matches"]
    assert [row["student_step"] for row in known] == [0, 1]
    ambiguous = known[1]
    assert ambiguous["teacher_steps"] == [1, 2]
    assert ambiguous["labels_consistent"] is False
    assert "action_error" not in ambiguous and "teacher_action" not in ambiguous
    assert "desired_translation_delta_error_m" not in ambiguous


def test_should_keep_all_fifty_student_steps_and_saturated_expert_history():
    study, packet = records(50)
    result = diagnosis.diagnose_push_control(study, packet)
    assert result["complete"] is True
    assert result["comparisons"]["corrections"]["rows"] == dict(teacher=50, student=50)
    assert len(result["comparisons"]["corrections"]["exact_observation_matches"]) == 50
    assert result["work"]["identity_copy_rows"] == 168
    assert result["work"]["taskspace12_distance_coordinates"] == 40800
    assert all(row["observation"][11] == 1.0 for row in study["training"]["transitions"][6:])


@pytest.mark.parametrize(
    "fault",
    [
        "incomplete",
        "unknown",
        "source",
        "seed",
        "role",
        "context",
        "spec",
        "recipe",
        "prediction",
        "work",
        "history",
        "packet_role",
        "packet_count",
        "reset",
        "prior_counts",
        "prior_unknown",
        "last_history",
    ],
)
def test_should_refuse_partial_rebound_or_noncausal_records_before_numeric_description(
    monkeypatch, fault
):
    study, packet = records()
    rollout = study["native_push"]
    if fault == "incomplete":
        study["complete"] = False
    elif fault == "unknown":
        rollout["unreturned_native_work_unknown"] = True
    elif fault == "source":
        study["source_unchanged"] = False
    elif fault == "prior_counts":
        study["correction_continuation"]["final_counts"]["training_rows"] += 1
    elif fault == "prior_unknown":
        study["correction_continuation"]["unreturned_training_work_unknown"] = True
    elif fault == "seed":
        rollout["metadata"]["seed"] = True
    elif fault == "role":
        rollout["metadata"]["role"] = "probe_test"
    elif fault == "context":
        rollout["metadata"]["context"] = "push-cube-train-19"
    elif fault == "spec":
        rollout["episode"]["spec"]["action_units"] = ("other",) + tuple(
            rollout["episode"]["spec"]["action_units"][1:]
        )
    elif fault == "recipe":
        rollout["metadata"]["recipe"]["history"] = "zero"
    elif fault == "prediction":
        rollout["prediction_returns"] -= 1
    elif fault == "work":
        rollout["work"]["completed_control_physics_steps"] += 1
    elif fault == "history":
        row = rollout["transitions"][1]
        row["observation"] = tuple(row["observation"][:7]) + (0.0,) * 5
        rollout["episode"]["transitions"][1] = deepcopy(row)
    elif fault == "last_history":
        row = rollout["transitions"][-1]
        row["next_observation"] = tuple(row["next_observation"][:7]) + (-0.5, 0.0, 0.0, -1.0, 1 / 6)
        rollout["episode"]["transitions"][-1] = deepcopy(row)
    elif fault == "packet_role":
        packet = replace(packet, role="probe_train")
    elif fault == "packet_count":
        short = packet.transitions[:-2] + (replace(packet.transitions[-2], terminated=True),)
        packet = replace(packet, transitions=short)
    else:
        row = rollout["transitions"][0]
        row["observation"] = (0.0, 0.3) + tuple(row["observation"][2:])
        rollout["episode"]["transitions"][0] = deepcopy(row)

    def forbidden(*args, **options):
        pytest.fail("inadmissible saved Push records reached numeric description")

    monkeypatch.setattr(diagnosis, "_describe_control", forbidden)
    with pytest.raises((ValueError, RuntimeError)):
        diagnosis.diagnose_push_control(study, packet)


def test_should_refuse_finite_but_overflowing_support_arithmetic():
    study, packet = records()
    for field in ("episode",):
        row = study["native_push"][field]["transitions"][1]
        row["observation"] = (0.0, 1e308) + tuple(row["observation"][2:])
        study["native_push"][field]["transitions"][0]["next_observation"] = row["observation"]
    study["native_push"]["transitions"] = deepcopy(study["native_push"]["episode"]["transitions"])
    with pytest.raises(ValueError, match="finite"):
        diagnosis.diagnose_push_control(study, packet)


def test_should_route_selected_cli_receipts_to_the_real_push_diagnosis(monkeypatch, capsys):
    study, packet = records()
    receipt = dict(kind="synthetic bound correction receipt", rows=50, path="corrections.json")
    study["correction_fit_protocol"]["input_receipt"] = receipt.copy()
    seen: list[tuple[str, Path]] = []

    def read_study(path):
        seen.append(("study", path))
        return study

    def read_packet(path):
        seen.append(("corrections", path))
        return packet, receipt

    monkeypatch.setattr(diagnosis, "_read_push_study", read_study)
    monkeypatch.setattr(diagnosis, "load_push_correction_inputs", read_packet)
    monkeypatch.setattr(
        diagnosis, "_read", lambda path: pytest.fail("selected Push CLI used legacy reader")
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "diagnosis",
            "--teacher",
            "corrections.json",
            "--learned",
            "balanced.json",
            "--push-taskspace",
        ],
    )
    diagnosis.main()
    report = json.loads(capsys.readouterr().out)
    assert report["complete"] is True
    assert report["work"]["identity_copy_rows"] == 74
    assert report["work"]["taskspace12_distance_coordinates"] == 2448
    assert seen == [("study", Path("balanced.json")), ("corrections", Path("corrections.json"))]


def test_should_preserve_default_cli_reader_and_legacy_call_shape(monkeypatch, capsys):
    teacher, learned = {"fixture": "teacher"}, {"fixture": "learned"}
    seen: list[Any] = []

    def read(path):
        seen.append(path)
        return teacher if path == Path("teacher.json") else learned

    def describe(*args, **options):
        assert args == (teacher, learned) and options == {}
        return dict(complete=True, legacy=True)

    monkeypatch.setattr(diagnosis, "_read", read)
    monkeypatch.setattr(diagnosis, "diagnose_control", describe)
    monkeypatch.setattr(
        diagnosis, "_read_push_study", lambda path: pytest.fail("default CLI read Push metadata")
    )
    monkeypatch.setattr(
        sys, "argv", ["diagnosis", "--teacher", "teacher.json", "--learned", "learned.json"]
    )
    diagnosis.main()
    assert json.loads(capsys.readouterr().out) == dict(complete=True, legacy=True)
    assert seen == [Path("teacher.json"), Path("learned.json")]


def test_should_refuse_combined_cli_modes_before_reading_any_record(monkeypatch):
    def forbidden(*args, **options):
        pytest.fail("combined CLI modes reached saved inputs")

    monkeypatch.setattr(diagnosis, "_read", forbidden)
    monkeypatch.setattr(diagnosis, "_read_push_study", forbidden)
    monkeypatch.setattr(diagnosis, "load_push_correction_inputs", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "diagnosis",
            "--teacher",
            "a",
            "--learned",
            "b",
            "--push-taskspace",
            "--state-observation",
        ],
    )
    with pytest.raises(SystemExit) as caught:
        diagnosis.main()
    assert caught.value.code == 2


def test_should_refuse_cli_receipt_replacement_before_numeric_description(monkeypatch):
    study, packet = records()
    study["correction_fit_protocol"]["input_receipt"] = dict(rows=50, bound="original", path="a")
    monkeypatch.setattr(diagnosis, "_read_push_study", lambda path: study)
    monkeypatch.setattr(
        diagnosis,
        "load_push_correction_inputs",
        lambda path: (packet, dict(rows=50, bound="replaced", path="a")),
    )
    monkeypatch.setattr(
        diagnosis,
        "diagnose_push_control",
        lambda *args: pytest.fail("replaced receipt reached diagnosis"),
    )
    monkeypatch.setattr(
        sys, "argv", ["diagnosis", "--teacher", "a", "--learned", "b", "--push-taskspace"]
    )
    with pytest.raises(ValueError, match="receipt"):
        diagnosis.main()


@pytest.mark.parametrize("fault", [None, "path", "other_value"])
def test_should_resolve_only_receipt_path_spelling_and_bind_every_other_value(tmp_path, fault):
    original: dict[str, Any] = dict(
        path=str(tmp_path / "corrections.json"), rows=50, bytes=128, sha256="synthetic"
    )
    selected: dict[str, Any] = dict(
        original, path=str(tmp_path / "sub" / ".." / "corrections.json")
    )
    if fault == "path":
        selected["path"] = str(tmp_path / "different.json")
    elif fault == "other_value":
        selected["bytes"] += 1
    before = deepcopy(original), deepcopy(selected)
    assert diagnosis._same_push_receipt(selected, original) is (fault is None)
    assert (original, selected) == before
