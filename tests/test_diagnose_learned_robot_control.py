"""Saved-data diagnosis, never a counterfactual teacher or policy experiment.

Prospective per-platform work: 3 successful 3-teacher/4-student diagnoses and
1 reset-refusal call (before normalization) and 1 finite-input distance-overflow
refusal. Together calls normalize 28 rows/22,092 coordinates, attempt 1,008
prefix coordinates and perform 28,404 exact coordinate comparisons.
Each successful selected working-payload bound is 152,760 bytes on the 64-bit
platforms. Other cases call pure admission or stub the diagnostic. Zero models,
fits, RNG, optimizer updates, forwards, gradients, or simulator controls.
"""

from copy import deepcopy
import io
import json
from pathlib import Path
import sys
from typing import Any

import pytest

from scripts import diagnose_learned_robot_control as diagnostic


UNITS = ["normalized_root_dx", "normalized_root_dy", "normalized_root_dz", "normalized_gripper"]


def observation(position: float, *, pixel: float = 0.25) -> list[float]:
    return [position] + [0.0] * 20 + [pixel] * 768


def episode(
    identity: str, observations: list[list[float]], actions: list[list[float]]
) -> dict[str, Any]:
    return {
        "episode_id": identity,
        "context": "pick-cube",
        "role": "train",
        "spec": {
            "observation_dim": 789,
            "action_units": UNITS.copy(),
            "lower_bounds": [-1.0] * 4,
            "upper_bounds": [1.0] * 4,
        },
        "transitions": [
            {
                "step": index,
                "observation": row.copy(),
                "action": action.copy(),
                "reward": 0.0,
                "next_observation": (
                    observations[index + 1].copy()
                    if index + 1 < len(observations)
                    else observation(5.0)
                ),
                "terminated": False,
                "truncated": index == len(observations) - 1,
            }
            for index, (row, action) in enumerate(zip(observations, actions, strict=True))
        ],
    }


def inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    warm = episode(
        "pick-cube-task-teacher-17",
        [observation(x) for x in (0.0, 2.0, 4.0)],
        [[0.5, -0.5, 0.0, -1.0], [-0.5, 0.5, 0.5, 1.0], [0.0, 0.0, 1.0, -1.0]],
    )
    student = episode(
        "pick-cube-smoke-17",
        [observation(0.0), observation(1.0), observation(4.0), observation(4.0, pixel=0.5)],
        [
            [0.6, -0.4, 0.25, 1.0],
            [1.0, -1.0, -1.0, -1.0],
            [0.0, 0.5, 0.75, -1.0],
            [-1.0, -1.0, -1.0, 1.0],
        ],
    )
    policy = {
        "spec": deepcopy(warm["spec"]),
        "hidden_dim": 64,
        "seed": 17,
        "learning_rate": 0.01,
        "momentum": 0.9,
        "training_role": "train",
        "observation_standardization": {
            "means": [0.0] * 789,
            "scales": [2.0] + [1.0] * 788,
            "fitted_rows": 3,
        },
        "binary_gripper_axis": None,
    }
    teacher = {
        "qualified": True,
        "role": "train",
        "seed": 17,
        "task": "PickCube-v1",
        "controller": "pd_ee_delta_pos",
        "context": "pick-cube",
        "candidate_episode": warm,
    }
    learned = {
        "complete": True,
        "candidate_unchanged": True,
        "training": {
            "complete": True,
            "policy": policy,
            "work": {"optimizer_updates": 24, "training_rows": 704},
            "source_episodes": {"warm": {"episode_id": warm["episode_id"]}},
        },
        "rollout": {
            "complete": True,
            "role": "train",
            "seed": 17,
            "context": "pick-cube",
            "controller": "pd_ee_delta_pos",
            "optimizer_updates": 0,
            "frozen_state_exact": True,
            "original_owner_exact": True,
            "learner_policy": deepcopy(policy),
            "source_updates": 24,
            "source_training_rows": 704,
            "episode": student,
            "sparse_return": 0.0,
            "any_success": False,
            "final_success": False,
        },
    }
    return teacher, learned


@pytest.fixture(scope="module")
def completed() -> dict[str, Any]:
    teacher, learned = inputs()
    before = deepcopy((teacher, learned))
    report = diagnostic.diagnose_control(teacher, learned)
    assert (teacher, learned) == before
    return report


def test_should_score_only_exact_full_observations(completed: dict[str, Any]) -> None:
    matches = completed["exact_observation_matches"]
    assert [row["student_step"] for row in matches] == [0, 2]
    assert [row["teacher_steps"] for row in matches] == [[0], [2]]
    assert all(row["labels_consistent"] is True for row in matches)
    assert matches[0]["action_error"] == pytest.approx([0.1, 0.1, 0.25, 2.0])
    assert matches[0]["max_abs_action_error"] == 2.0
    assert matches[0]["desired_translation_delta_error_m"] == pytest.approx([0.01, 0.01, 0.025])
    assert matches[1]["action_error"] == pytest.approx([0.0, 0.5, -0.25, 0.0])
    assert matches[1]["max_abs_action_error"] == 0.5


def test_should_describe_nearest_support_without_using_it_as_labels(
    completed: dict[str, Any],
) -> None:
    assert completed["nearest_public21_teacher_steps"] == [0, 0, 2, 2]
    assert completed["nearest_public21_normalized_rms"] == pytest.approx(
        [0.0, 0.5 / (21**0.5), 0.0, 0.0]
    )
    assert completed["first_nonzero_prefix_support_distance"] == 1
    assert completed["observation_blocks"] == {
        "public21": {
            "outside_training_range_coordinates_by_step": [0, 0, 0, 0],
            "max_abs_frozen_normalized_by_step": [0.0, 0.5, 2.0, 2.0],
        },
        "rgb768": {
            "outside_training_range_coordinates_by_step": [0, 0, 0, 768],
            "max_abs_frozen_normalized_by_step": [0.25, 0.25, 0.25, 0.5],
        },
    }
    assert completed["recorded_outcome"] == {
        "sparse_return": 0.0,
        "any_success": False,
        "final_success": False,
    }


def test_should_account_for_selected_payload_and_no_model_work(completed: dict[str, Any]) -> None:
    assert completed["rows"] == {"teacher": 3, "student": 4}
    assert completed["work"] == {
        "normalization_calls": 2,
        "normalization_rows": 7,
        "normalization_coordinates": 5523,
        "prefix_distance_coordinates": 252,
        "exact_match_coordinate_comparisons": 9468,
        "selected_numeric_boolean_bytes": 90780,
        "selected_temporary_allowance_bytes": 61980,
        "selected_working_payload_bound_bytes": 152760,
        "model_initializations": 0,
        "fits": 0,
        "optimizer_updates": 0,
        "forwards": 0,
        "gradients": 0,
        "rng_draws": 0,
        "simulator_controls": 0,
    }
    assert len(json.dumps(completed, allow_nan=False).encode("utf8")) < 65536


def test_should_not_inherit_time_or_nearest_labels_off_trajectory(
    completed: dict[str, Any],
) -> None:
    teacher, learned = inputs()
    teacher["candidate_episode"]["transitions"][1]["action"] = [1.0, -1.0, -1.0, -1.0]
    # This row is nearest/tied to an off-trajectory input and occupies its time
    # index, but has no exactly equal student observation. Its label is irrelevant.
    assert diagnostic.diagnose_control(teacher, learned) == completed


def test_should_leave_conflicting_duplicate_observations_unscored() -> None:
    teacher, learned = inputs()
    rows = teacher["candidate_episode"]["transitions"]
    rows[1]["observation"] = rows[0]["observation"].copy()
    rows[0]["next_observation"] = rows[1]["observation"].copy()
    report = diagnostic.diagnose_control(teacher, learned)
    ambiguous = report["exact_observation_matches"][0]
    assert ambiguous == {"student_step": 0, "teacher_steps": [0, 1], "labels_consistent": False}
    assert report["exact_observation_matches"][1]["student_step"] == 2
    assert "action_error" in report["exact_observation_matches"][1]


def test_should_refuse_different_reset_before_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    teacher, learned = inputs()
    learned["rollout"]["episode"]["transitions"][0]["observation"][0] = 0.01

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("reset mismatch reached frozen normalization")

    monkeypatch.setattr(diagnostic, "standardize_observations", forbidden)
    with pytest.raises(ValueError, match="reset"):
        diagnostic.diagnose_control(teacher, learned)


def test_should_refuse_finite_input_whose_distance_square_overflows() -> None:
    teacher, learned = inputs()
    policy = learned["training"]["policy"]
    policy["observation_standardization"]["scales"] = [1.0] * 789
    learned["rollout"]["learner_policy"] = deepcopy(policy)
    rows = learned["rollout"]["episode"]["transitions"]
    rows[1]["observation"][0] = 1e155
    rows[0]["next_observation"] = rows[1]["observation"].copy()
    before = deepcopy((teacher, learned))
    with pytest.raises(ValueError, match="finite"):
        diagnostic.diagnose_control(teacher, learned)
    assert (teacher, learned) == before


@pytest.mark.parametrize(
    "path,value",
    [
        (("teacher", "qualified"), False),
        (("teacher", "role"), "final_test"),
        (("teacher", "seed"), 19),
        (("learned", "complete"), False),
        (("teacher", "seed"), 17.0),
        (("learned", "candidate_unchanged"), False),
        (("learned", "training", "complete"), False),
        (("learned", "rollout", "complete"), False),
        (("learned", "rollout", "role"), "probe_test"),
        (("learned", "rollout", "seed"), 19),
        (("learned", "rollout", "seed"), 17.0),
        (("learned", "rollout", "controller"), "pd_joint_pos"),
        (("learned", "rollout", "optimizer_updates"), 1),
        (("learned", "rollout", "optimizer_updates"), False),
        (("learned", "rollout", "optimizer_updates"), 0.0),
        (("learned", "rollout", "frozen_state_exact"), False),
        (("learned", "rollout", "original_owner_exact"), None),
        (("learned", "rollout", "source_updates"), 23),
        (("learned", "rollout", "source_updates"), 24.0),
        (("learned", "rollout", "source_training_rows"), 703),
        (("learned", "rollout", "source_training_rows"), 704.0),
        (("learned", "training", "work", "optimizer_updates"), 24.0),
        (("learned", "training", "work", "training_rows"), 704.0),
        (("learned", "training", "source_episodes", "warm", "episode_id"), "another-teacher"),
        (("teacher", "candidate_episode", "role"), "inner_guard"),
        (("learned", "rollout", "episode", "role"), "probe_train"),
        (("learned", "rollout", "episode", "context"), "another-context"),
    ],
)
def test_should_refuse_unbound_saved_authority_before_array_math(
    path: tuple[str, ...], value: Any
) -> None:
    teacher, learned = inputs()
    roots = {"teacher": teacher, "learned": learned}
    target = roots[path[0]]
    for key in path[1:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        diagnostic._admit(teacher, learned)


@pytest.mark.parametrize(
    "damage",
    ["different_policy", "zero_scale", "nonfinite_mean", "wrong_fit_rows", "short_statistics"],
)
def test_should_refuse_invalid_or_unbound_frozen_statistics(damage: str) -> None:
    teacher, learned = inputs()
    policy = learned["training"]["policy"]
    if damage == "different_policy":
        learned["rollout"]["learner_policy"]["momentum"] = 0.8
    else:
        transform = policy["observation_standardization"]
        if damage == "zero_scale":
            transform["scales"][0] = 0.0
        elif damage == "nonfinite_mean":
            transform["means"][0] = float("nan")
        elif damage == "short_statistics":
            transform["means"] = [0.0] * 788
            transform["scales"] = [1.0] * 788
        else:
            transform["fitted_rows"] = 2
        learned["rollout"]["learner_policy"] = deepcopy(policy)
    with pytest.raises(ValueError):
        diagnostic._admit(teacher, learned)


def test_should_own_admitted_episode_and_statistics() -> None:
    teacher, learned = inputs()
    warm, student, transform = diagnostic._admit(teacher, learned)
    teacher["candidate_episode"]["transitions"][0]["action"][0] = -1.0
    learned["rollout"]["episode"]["transitions"][0]["observation"][0] = 99.0
    learned["training"]["policy"]["observation_standardization"]["means"][0] = 99.0
    assert warm.transitions[0].action[0] == 0.5
    assert student.transitions[0].observation[0] == 0.0
    assert transform.means[0] == 0.0


def test_should_bound_input_to_one_owned_read_before_json(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int | None] = []

    class Payload(io.BytesIO):
        def read(self, size: int | None = -1) -> bytes:
            calls.append(size)
            return super().read(size)

    def open_payload(self: Path, mode: str) -> Payload:
        assert mode == "rb"
        return Payload(b" " * (8 * 1024**2 + 1))

    monkeypatch.setattr(Path, "open", open_payload)
    with pytest.raises(ValueError, match="8MiB"):
        diagnostic._read(Path("oversized.json"))
    assert calls == [8 * 1024**2 + 1]


def test_should_route_cli_files_without_extra_diagnostic_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    teacher, learned = inputs()
    teacher_path, learned_path = tmp_path / "teacher.json", tmp_path / "learned.json"
    teacher_path.write_text(json.dumps(teacher), encoding="utf8")
    learned_path.write_text(json.dumps(learned), encoding="utf8")
    calls: list[tuple[dict[str, Any], dict[str, Any]]] = []

    def stub(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
        calls.append((first, second))
        return {"complete": True, "recorded_outcome": {"final_success": False}}

    monkeypatch.setattr(diagnostic, "diagnose_control", stub)
    monkeypatch.setattr(
        sys, "argv", ["diagnose", "--teacher", str(teacher_path), "--learned", str(learned_path)]
    )
    diagnostic.main()
    assert calls == [(teacher, learned)]
    assert json.loads(capsys.readouterr().out) == {
        "complete": True,
        "recorded_outcome": {"final_success": False},
    }


@pytest.mark.parametrize(
    "result,match", [({"data": "x" * 65536}, "64KiB"), ({"data": float("nan")}, "Out of range")]
)
def test_should_refuse_nonfinite_or_oversized_output_before_print(
    result: dict[str, Any],
    match: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(diagnostic, "_read", lambda path: {})
    monkeypatch.setattr(diagnostic, "diagnose_control", lambda *args: result)
    monkeypatch.setattr(sys, "argv", ["diagnose", "--teacher", "a", "--learned", "b"])
    with pytest.raises(ValueError, match=match):
        diagnostic.main()
    assert capsys.readouterr().out == ""
