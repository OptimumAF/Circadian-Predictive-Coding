"""Metadata/stub coverage for the fixed cumulative native context composition.

No arrays, fitting, models, RNG, predictions, appends, or simulator calls. Three
run_case paths use stub readers/collector/stream/readouts. The successful path
returns8 fake readouts; retained-replacement path returns1; failed teacher0.
Immutable two-row core episodes validate real finite trace metadata only.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass
import importlib
import json
from types import SimpleNamespace
from typing import Any

import pytest

from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)
from src.core.pick_cube_lighting_context import pick_cube_lighting_context
from src.core.pick_cube_mass_context import pick_cube_mass_context

case = importlib.import_module("artifacts.runs.continuous-context-sequence-20261020.native_case")


def teacher(context: str = "pick-cube") -> ContinuousEpisode:
    spec = ContinuousActionSpec(
        789,
        ("normalized_root_dx", "normalized_root_dy", "normalized_root_dz", "normalized_gripper"),
        (-1.0,) * 4,
        (1.0,) * 4,
    )
    observations = [(float(index),) + (0.0,) * 20 + (0.25,) * 768 for index in range(3)]
    return ContinuousEpisode(
        "qualified-teacher",
        context,
        "train",
        spec,
        tuple(
            ContinuousTransition(
                index,
                observations[index],
                (0.0, 0.0, 0.0, 1.0),
                0.0,
                observations[index + 1],
                index == 1,
                False,
            )
            for index in range(2)
        ),
    )


def readback(name: str) -> dict[str, Any]:
    if name == "mass_context":
        return {
            "context": asdict(pick_cube_mass_context()),
            "source_mass_kg": 0.125,
            "applied_mass_kg": 0.25,
        }
    return {
        "context": asdict(pick_cube_lighting_context()),
        "source_ambient_rgb": [0.5] * 3,
        "applied_ambient_rgb": [0.25] * 3,
    }


def trace(record=None, *, success: bool = False, options=None) -> dict[str, Any]:
    record = teacher() if record is None else record
    episode = json.loads(json.dumps(asdict(record)))
    episode["episode_id"] = "pick-cube-smoke-17"
    episode["transitions"][-1]["reward"] = float(success)
    rows = len(episode["transitions"])
    value: dict[str, Any] = {
        "complete": True,
        "error": None,
        "seed": 17,
        "role": "train",
        "context": record.context,
        "episode": episode,
        "visual_recipe_exact": True,
        "sparse_return": float(success),
        "final_success": success,
        "any_success": success,
        "joint_position_exact": True,
        "joint_position_summary": {
            "limits": {
                "names": [f"panda_joint{i}" for i in range(1, 8)]
                + ["panda_finger_joint1", "panda_finger_joint2"],
                "units": ["rad"] * 7 + ["m"] * 2,
                "lower": [-3.0] * 7 + [0.0] * 2,
                "upper": [3.0] * 7 + [0.04] * 2,
            },
            "samples": rows + 1,
            "violating_samples": 0,
            "lower_peak": [0.0] * 9,
            "upper_peak": [0.0] * 9,
            "violations_per_joint": [0] * 9,
        },
        "joint_position_work": {
            "limit_read_attempts": 1,
            "limit_read_returns": 1,
            "samples": rows + 1,
            "sampled_coordinates": (rows + 1) * 9,
            "name_read_attempts": 1,
            "name_read_returns": 1,
        },
        "frozen_state_exact": True,
        "original_owner_exact": True,
        "work": {
            "step_attempts": rows,
            "completed_steps": rows,
            "completed_control_physics_steps": rows * 5,
            "possible_failed_physics_steps": 0,
        },
        "visual_work": {
            "observation_attempts": rows + 1,
            "observation_returns": rows + 1,
            "input_frame_bytes_upper_bound": (rows + 1) * 49152,
            "feature_values_upper_bound": (rows + 1) * 789,
            "returned_frame_bytes": (rows + 1) * 49152,
            "returned_feature_values": (rows + 1) * 789,
            "returned_feature_numeric_bytes": (rows + 1) * 6312,
            "current_owned_frame_bytes": 49152,
            "frame_export_copies": 0,
            "frame_export_bytes": 0,
        },
        "prediction_attempts": rows,
        "prediction_returns": rows,
        "nearest_distance_candidates_completed": rows * 29,
        "nearest_distance_feature_values_completed": rows * 29 * 789,
    }
    for name, context in (options or {}).items():
        value[name] = asdict(context)
        value[name + "_exact"] = True
        value[name.replace("_context", "_readback")] = readback(name)
    return value


def new_audit() -> dict[str, Any]:
    return {
        "complete": False,
        "teacher_attempts": 0,
        "teacher_returns": 0,
        "readouts": [],
        "source_rows": [],
    }


def test_should_accept_complete_negative_endpoints_and_preserve_costs_without_modifying_inputs():
    record, raw = teacher(), trace()
    before = deepcopy(raw)
    summary = case.validate_rollout(raw, deepcopy(raw), record, {})
    assert summary["rows"] == 2
    assert summary["sparse_return"] == 0.0
    assert summary["final_success"] is summary["any_success"] is False
    assert summary["prediction_returns"] == 2
    assert summary["distance_candidates"] == 58 and summary["distance_feature_values"] == 45762
    assert summary["joint_position_summary"]["samples"] == 3
    assert raw == before


@pytest.mark.parametrize(
    "damage",
    [
        "retained",
        "seed",
        "role",
        "context",
        "pixel",
        "endpoint",
        "joint",
        "joint_samples",
        "owner",
        "work",
        "intervention",
    ],
)
def test_should_refuse_malformed_or_replaced_native_evidence(damage):
    record, raw = teacher(), trace()
    options: dict[str, Any] = {}
    if damage == "retained":
        returned = deepcopy(raw)
        raw["work"]["completed_control_physics_steps"] = 11
    else:
        if damage == "seed":
            raw["seed"] = 17.0
        elif damage == "role":
            raw["role"] = "final_test"
        elif damage == "context":
            raw["context"] = "another-context"
        elif damage == "pixel":
            raw["episode"]["transitions"][-1]["next_observation"][21] = -0.01
        elif damage == "endpoint":
            raw["any_success"] = True
        elif damage == "joint":
            raw["joint_position_exact"] = False
        elif damage == "joint_samples":
            raw["joint_position_summary"]["samples"] = 2
        elif damage == "owner":
            raw["frozen_state_exact"] = False
        elif damage == "work":
            raw["work"]["completed_control_physics_steps"] = 11
        else:
            options = {"mass_context": pick_cube_mass_context()}
            raw["mass_context"] = {"context": "another-mass"}
            raw["mass_context_exact"] = True
        returned = deepcopy(raw)
    with pytest.raises(ValueError):
        case.validate_rollout(raw, returned, record, options)


@dataclass(frozen=True)
class CollectedTeacher:
    qualified: bool = True


@pytest.fixture
def composition(monkeypatch, tmp_path):
    spec = teacher().spec
    warm = SimpleNamespace(context="pick-cube", spec=spec, transitions=(None,) * 29)
    mass = SimpleNamespace(
        context=pick_cube_mass_context().context, spec=spec, transitions=(None,) * 2
    )
    lighting = SimpleNamespace(
        context=pick_cube_lighting_context().context, spec=spec, transitions=(None,) * 29
    )
    calls: dict[str, Any] = {"collect": [], "stream": [], "native": [], "mass_reads": []}
    collected = {"value": CollectedTeacher()}

    def collect(**options):
        calls["collect"].append(options)
        return collected["value"]

    def read_mass(path, **options):
        calls["mass_reads"].append((path, options))
        return mass, {"mode": "mass_drift", "mass_readback": readback("mass_context")}

    def rollout(path, **options):
        calls["native"].append((path, options))
        context_options = {
            key: options[key] for key in ("mass_context", "lighting_context") if key in options
        }
        context = next(iter(context_options.values())).context if context_options else "pick-cube"
        raw = trace(teacher(context), options=context_options)
        case.write_new(path, raw)
        return SimpleNamespace(report=raw)

    def stream(first, second, *, additional_episodes, checkpoint_observer):
        calls["stream"].append((first, second, additional_episodes))
        results = []
        for phase, index in (
            ("warm", None),
            ("before_release", 0),
            ("after_release", 0),
            ("before_release", 1),
            ("after_release", 1),
        ):
            results.append(checkpoint_observer(object(), {"phase": phase, "packet_index": index}))
        return {
            "complete": True,
            "checkpoints": results,
            "applied": [{"update_number": 1}, {"update_number": 2}],
        }

    monkeypatch.setattr(case, "STAGE", tmp_path)
    monkeypatch.setattr(case, "_read_teacher", lambda *args, **kwargs: warm)
    monkeypatch.setattr(
        case,
        "_read_lighting_teacher",
        lambda *args, **kwargs: (
            lighting,
            {"mode": "lighting_drift", "lighting_readback": readback("lighting_context")},
        ),
    )
    monkeypatch.setattr(case, "_read_mass_teacher", read_mass)
    monkeypatch.setattr(case, "collect_task_teacher", collect)
    monkeypatch.setattr(case.smoke, "run_frozen_rollout", rollout)
    monkeypatch.setattr(case, "run_continuous_demo", stream)
    monkeypatch.setattr(case, "checkpoint_state", lambda readout: None)
    monkeypatch.setattr(case, "_owned_state", lambda readout, state: None)
    return tmp_path, calls, collected, (warm, mass, lighting)


def test_should_compose_exact_fixed_schedule_and_accept_all_negative_readouts(composition):
    directory, calls, _, records = composition
    audit = new_audit()
    case.run_case(audit)
    assert audit["complete"] is True
    assert audit["teacher_attempts"] == audit["teacher_returns"] == 1
    assert audit["source_rows"] == [29, 2, 29]
    assert calls["stream"] == [(records[0], records[1], (records[2],))]
    assert [row["target"] for row in audit["readouts"]] == [
        "A",
        "mass",
        "mass",
        "A",
        "lighting",
        "lighting",
        "mass",
        "A",
    ]
    assert all(
        row["attempted"] is row["returned"] is row["validated"] is True for row in audit["readouts"]
    )
    assert all(row["summary"]["sparse_return"] == 0.0 for row in audit["readouts"])
    assert all(
        options["train_seed"] == 17
        and options["visual_observation"] is True
        and options["monitor_joint_positions"] is True
        for _, options in calls["native"]
    )
    assert len({id(options["learner"]) for _, options in calls["native"]}) == 5
    assert (directory / "teacher-mass-rgb.json").exists()
    assert len(list(directory.glob("readout-*.json"))) == 8


def test_should_preserve_returned_unqualified_teacher_and_stop_before_stream(composition):
    directory, calls, collected, _ = composition
    collected["value"] = CollectedTeacher(False)
    audit = new_audit()
    with pytest.raises(ValueError, match="did not qualify"):
        case.run_case(audit)
    assert audit["teacher_attempts"] == audit["teacher_returns"] == 1
    assert calls["stream"] == calls["native"] == calls["mass_reads"] == []
    assert (
        json.loads((directory / "teacher-mass-rgb.json").read_text(encoding="utf8"))["qualified"]
        is False
    )


def test_should_refuse_wrong_saved_teacher_size_before_collecting_new_labels(composition):
    _, calls, _, records = composition
    records[2].transitions = (None,) * 30
    audit = new_audit()
    with pytest.raises(ValueError):
        case.run_case(audit)
    assert calls["collect"] == calls["mass_reads"] == calls["stream"] == calls["native"] == []
    assert audit["teacher_attempts"] == audit["teacher_returns"] == 0


def test_should_refuse_changed_preceding_readout_before_more_native_work(composition, monkeypatch):
    directory, calls, _, _ = composition

    def stream(warm, mass, *, additional_episodes, checkpoint_observer):
        calls["stream"].append((warm, mass, additional_episodes))
        checkpoint_observer(object(), {"phase": "warm", "packet_index": None})
        path = directory / "readout-0-A.json"
        changed = case.read_bounded(path)
        changed["sparse_return"] = 99.0
        path.write_text(json.dumps(changed), encoding="utf8")
        checkpoint_observer(object(), {"phase": "before_release", "packet_index": 0})
        pytest.fail("replaced preceding readout admitted further work")

    monkeypatch.setattr(case, "run_continuous_demo", stream)
    audit = new_audit()
    with pytest.raises(ValueError, match="preceding native receipt changed"):
        case.run_case(audit)
    assert audit["complete"] is False
    assert len(calls["native"]) == len(audit["readouts"]) == 1
    assert audit["readouts"][0]["validated"] is True


def test_should_keep_partial_paid_audit_when_main_catches_failure(monkeypatch, tmp_path):
    def failed(audit):
        audit["teacher_attempts"] = audit["teacher_returns"] = 1
        audit["readouts"].append({"attempted": True, "returned": True, "validated": False})
        raise RuntimeError("synthetic returned native validation failure")

    monkeypatch.setattr(case, "STAGE", tmp_path)
    monkeypatch.setattr(case, "run_case", failed)
    monkeypatch.setattr(case, "preflight", lambda: None)
    monkeypatch.setattr(case, "peak_rss_bytes", lambda: 1)
    monkeypatch.setenv("VK_ICD_FILENAMES", "stub-software-vulkan")
    assert case.main() == 1
    report = json.loads((tmp_path / "native-result.json").read_text(encoding="utf8"))
    assert report["complete"] is False
    assert report["teacher_returns"] == 1 and report["readouts"][0]["returned"] is True
    assert "synthetic returned native validation failure" in report["error"]


def test_should_keep_paid_readout_but_refuse_late_native_state_change(composition, monkeypatch):
    _, calls, _, _ = composition
    checks = []

    def guard(readout, state):
        checks.append(readout)
        if len(checks) == 2:
            raise ValueError("late readout state change")

    monkeypatch.setattr(case, "_owned_state", guard)
    audit = new_audit()
    with pytest.raises(ValueError, match="late readout state change"):
        case.run_case(audit)
    assert audit["complete"] is False
    assert len(calls["native"]) == 1
    record = audit["readouts"][0]
    assert record["returned"] is True and record["validated"] is False
    assert record["paid_return"]["work"]["completed_steps"] == 2
