"""Metadata-only composition of the missing cumulative camera boundary.

No arrays/models/fits/RNG/predictions/appends/native operations. Saved-reader,
stream, checkpoint-state and frozen-owner dependencies are explicit stubs; real
bounded JSON and core finite trace validation remain exercised. Per-platform
stub counts are frozen in the agent handoff before the root runs these tests.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass
import importlib
import json
from types import SimpleNamespace
from typing import Any

import pytest

from src.core.pick_cube_camera_context import camera_context_recipe, pick_cube_camera_context
from src.core.pick_cube_lighting_context import pick_cube_lighting_context
from src.core.pick_cube_mass_context import pick_cube_mass_context
from src.core.pick_cube_rgb_features import pick_cube_rgb_recipe
from test_continuous_context_case import readback, teacher, trace

case = importlib.import_module("artifacts.runs.camera-stream-input-20261020.camera_case")


@dataclass(frozen=True)
class FrozenPolicy:
    definition: str = "metadata-only original A29 policy"


def new_audit():
    return {"complete": False, "readouts": [], "source_rows": []}


def native_trace(context, options):
    raw = trace(
        teacher(context),
        options={key: value for key, value in options.items() if key != "camera_context"},
    )
    camera = options.get("camera_context")
    raw["observation_recipe"] = json.loads(
        json.dumps(
            asdict(pick_cube_rgb_recipe() if camera is None else camera_context_recipe(camera))
        )
    )
    if camera is not None:
        raw["camera_context"] = asdict(camera)
    return raw


@pytest.fixture
def composition(monkeypatch, tmp_path):
    contexts = (
        "pick-cube",
        pick_cube_mass_context().context,
        pick_cube_lighting_context().context,
        pick_cube_camera_context().context,
    )
    records = tuple(
        SimpleNamespace(
            context=context,
            role="train",
            spec=teacher(context).spec,
            episode_id=f"saved-{index}",
            transitions=(teacher(context).transitions[0],) * 29,
        )
        for index, context in enumerate(contexts)
    )
    calls: dict[str, Any] = {"readers": [], "stream": [], "native": [], "checks": [], "phases": []}
    metadata = {
        "mass": {"mass_readback": readback("mass_context")},
        "lighting": {"lighting_readback": readback("lighting_context")},
        "camera": {"camera_context": asdict(pick_cube_camera_context())},
    }

    def reader(index):
        def read(*args, **kwargs):
            calls["readers"].append((index, args, kwargs))
            return (
                records[index]
                if index == 0
                else (
                    records[index],
                    deepcopy(metadata[("A", "mass", "lighting", "camera")[index]]),
                )
            )

        return read

    bounded = case.prior.read_bounded

    def read(path):
        if path.name == "native-result.json":
            targets = ("A", "mass", "mass", "A", "lighting", "lighting", "mass", "A")
            phases = (
                ("warm", None, 0),
                ("before_release", 0, 0),
                ("after_release", 0, 2),
                ("after_release", 0, 2),
                ("before_release", 1, 3),
                ("after_release", 1, 5),
                ("after_release", 1, 5),
                ("after_release", 1, 5),
            )
            return {
                "complete": True,
                "stream": {"policy": asdict(FrozenPolicy())},
                "readouts": [
                    {
                        "path": f"readout-{index}-{target}.json",
                        "target": target,
                        "checkpoint": {"phase": phase, "packet_index": packet, "tick": tick},
                        "summary": {
                            "rows": 2,
                            "sparse_return": 0.0,
                            "any_success": False,
                            "final_success": False,
                            "joint_position_summary": {"violating_samples": 1},
                        },
                    }
                    for index, (target, (phase, packet, tick)) in enumerate(
                        zip(targets, phases, strict=True)
                    )
                ],
            }
        return bounded(path)

    def guard(readout, expected):
        calls["checks"].append((readout, expected))
        if readout.poisoned:
            raise ValueError("synthetic frozen native state changed")

    def rollout(path, **options):
        calls["native"].append((path, options))
        interventions = {
            key: options[key]
            for key in ("camera_context", "mass_context", "lighting_context")
            if key in options
        }
        context = next(iter(interventions.values())).context if interventions else "pick-cube"
        raw = native_trace(context, interventions)
        # Retained qpos samples are0,1,2 on joint0: the final sample exceeds1.5.
        # A valid negative run with observed constraint violations still reports.
        raw["joint_position_summary"]["limits"]["upper"][0] = 1.5
        raw["joint_position_summary"]["violating_samples"] = 1
        raw["joint_position_summary"]["violations_per_joint"][0] = 1
        raw["joint_position_summary"]["upper_peak"][0] = 0.5
        case.prior.write_new(path, raw)
        return SimpleNamespace(report=raw)

    def stream(warm, mass, *, additional_episodes, checkpoint_observer):
        calls["stream"].append((warm, mass, additional_episodes))
        results = []
        schedule = (
            ("warm", None, 0, 29, 1),
            ("before_release", 0, 0, 29, 1),
            ("after_release", 0, 2, 58, 2),
            ("before_release", 1, 3, 58, 2),
            ("after_release", 1, 5, 87, 3),
            ("before_release", 2, 6, 87, 3),
            ("after_release", 2, 8, 116, 4),
        )
        for phase, index, tick, rows, publications in schedule:
            readout = SimpleNamespace(
                poisoned=False,
                state=SimpleNamespace(
                    policy=FrozenPolicy(), training_examples=rows, completed_updates=publications
                ),
            )
            checkpoint = {
                "phase": phase,
                "packet_index": index,
                "tick": tick,
                "prototype_rows": rows,
                "completed_updates": publications,
            }
            before = len(calls["native"])
            result = checkpoint_observer(readout, checkpoint)
            calls["phases"].append((phase, index, before, len(calls["native"]), deepcopy(result)))
            results.append(result)
        return {
            "complete": True,
            "checkpoints": results,
            "experiment_state": {"completed_updates": 3, "next_tick": 9},
            "model_state": {
                "source_publications": 1,
                "candidate_publications": 4,
                "candidate_prototype_rows": 116,
            },
        }

    monkeypatch.setattr(case, "STAGE", tmp_path)
    for index, name in enumerate(
        ("_read_teacher", "_read_mass_teacher", "_read_lighting_teacher", "_read_camera_teacher")
    ):
        monkeypatch.setattr(case, name, reader(index))
    monkeypatch.setattr(case.prior, "read_bounded", read)
    monkeypatch.setattr(case.prior, "checkpoint_state", lambda readout: readout.state)
    monkeypatch.setattr(case, "_owned_state", guard)
    monkeypatch.setattr(case.smoke, "run_frozen_rollout", rollout)
    monkeypatch.setattr(case, "run_continuous_demo", stream)
    return tmp_path, records, calls


def test_should_request_native_work_only_at_camera_boundary_and_accept_negative_outcomes(
    composition,
):
    directory, records, calls = composition
    audit = new_audit()
    case.run_case(audit)
    assert audit["complete"] is True and audit["source_rows"] == [29] * 4
    assert len(audit["reused_native_points"]) == 8
    assert all(point["historical_not_new_work"] is True for point in audit["reused_native_points"])
    assert [point["target"] for point in audit["reused_native_points"]] == [
        "A",
        "mass",
        "mass",
        "A",
        "lighting",
        "lighting",
        "mass",
        "A",
    ]
    assert calls["stream"] == [(records[0], records[1], (records[2], records[3]))]
    assert [row["target"] for row in audit["readouts"]] == [
        "camera",
        "camera",
        "mass",
        "lighting",
        "A",
    ]
    assert all(row["returned"] is row["validated"] is True for row in audit["readouts"])
    assert all(row["summary"]["sparse_return"] == 0.0 for row in audit["readouts"])
    assert all(
        row["summary"]["joint_position_summary"]["violating_samples"] == 1
        for row in audit["readouts"]
    )
    assert [(row[2], row[3]) for row in calls["phases"][:5]] == [(0, 0)] * 5
    assert all(row[4] == {"readouts": [], "native_controls": 0} for row in calls["phases"][:5])
    assert [(row[2], row[3]) for row in calls["phases"][5:]] == [(0, 1), (1, 5)]
    assert all(
        options["train_seed"] == 17
        and options["monitor_joint_positions"] is True
        and options["visual_observation"] is True
        for _, options in calls["native"]
    )
    assert len(list(directory.glob("camera-readout-*.json"))) == 5


@pytest.mark.parametrize("index", [0, 3])
def test_should_refuse_changed_source_count_before_model_or_stream(composition, index):
    _, records, calls = composition
    records[index].transitions = records[index].transitions[:-1]
    with pytest.raises(ValueError, match="source29"):
        case.run_case(new_audit())
    assert calls["stream"] == calls["native"] == []


def test_should_refuse_wrong_preceding_publication_count_before_camera_work(
    composition, monkeypatch
):
    _, _, calls = composition

    def state(readout):
        readout.state.completed_updates = 2
        return readout.state

    monkeypatch.setattr(case.prior, "checkpoint_state", state)
    audit = new_audit()
    with pytest.raises(ValueError):
        case.run_case(audit)
    assert calls["native"] == audit["readouts"] == []
    assert len(calls["phases"]) == 5


@pytest.mark.parametrize("damage", ["recipe", "context", "reset"])
def test_should_refuse_camera_binding_or_qualified_start_mismatch(damage):
    config = pick_cube_camera_context()
    record = teacher(config.context)
    options = {"camera_context": config}
    raw = native_trace(config.context, options)
    if damage == "recipe":
        raw["observation_recipe"] = json.loads(json.dumps(asdict(pick_cube_rgb_recipe())))
    elif damage == "context":
        raw["camera_context"]["context"] = "another-camera"
    else:
        raw["episode"]["transitions"][0]["observation"][0] = 0.125
    with pytest.raises(ValueError):
        case.validate_target(raw, deepcopy(raw), record, options, {})


def test_should_refuse_replaced_first_camera_readout_before_any_return_native_work(
    composition, monkeypatch
):
    directory, _, calls = composition
    original = case.run_continuous_demo

    def stream(warm, mass, *, additional_episodes, checkpoint_observer):
        def observed(readout, checkpoint):
            if checkpoint["phase"] == "after_release" and checkpoint["packet_index"] == 2:
                path = directory / "camera-readout-0-camera.json"
                raw = case.prior.read_bounded(path)
                raw["sparse_return"] = 99.0
                path.write_text(json.dumps(raw), encoding="utf8")
            return checkpoint_observer(readout, checkpoint)

        return original(
            warm, mass, additional_episodes=additional_episodes, checkpoint_observer=observed
        )

    monkeypatch.setattr(case, "run_continuous_demo", stream)
    audit = new_audit()
    with pytest.raises(ValueError, match="preceding paid camera readout"):
        case.run_case(audit)
    assert len(calls["native"]) == len(audit["readouts"]) == 1
    assert audit["readouts"][0]["validated"] is True


def test_should_retain_paid_return_when_late_frozen_owner_check_fails(composition, monkeypatch):
    _, _, calls = composition
    original = case.smoke.run_frozen_rollout

    def rollout(path, **options):
        returned = original(path, **options)
        options["learner"].poisoned = True
        return returned

    monkeypatch.setattr(case.smoke, "run_frozen_rollout", rollout)
    audit = new_audit()
    with pytest.raises(ValueError, match="frozen native state changed"):
        case.run_case(audit)
    assert len(calls["native"]) == 1
    record = audit["readouts"][0]
    assert record["returned"] is True and record["validated"] is False
    assert record["paid_return"]["work"]["completed_steps"] == 2


def test_should_preserve_failed_native_paid_report_and_stop_later_returns(composition, monkeypatch):
    _, _, calls = composition
    original = case.smoke.run_frozen_rollout

    def rollout(path, **options):
        returned = original(path, **options)
        returned.report.update(complete=False, error="synthetic paid native failure")
        path.write_text(json.dumps(returned.report), encoding="utf8")
        return returned

    monkeypatch.setattr(case.smoke, "run_frozen_rollout", rollout)
    audit = new_audit()
    with pytest.raises(ValueError, match="incomplete"):
        case.run_case(audit)
    assert len(calls["native"]) == 1
    assert audit["readouts"][0]["paid_return"]["prediction_returns"] == 2
    assert audit["readouts"][0]["validated"] is False


@pytest.mark.parametrize(
    "target,key", [("mass", "mass_readback"), ("lighting", "lighting_readback")]
)
def test_should_stop_on_old_context_readback_mismatch_with_paid_report_retained(
    composition, monkeypatch, target, key
):
    _, _, calls = composition
    original = case.smoke.run_frozen_rollout

    def rollout(path, **options):
        returned = original(path, **options)
        if key in returned.report:
            returned.report[key]["context"]["seed"] = 19
            path.write_text(json.dumps(returned.report), encoding="utf8")
        return returned

    monkeypatch.setattr(case.smoke, "run_frozen_rollout", rollout)
    audit = new_audit()
    with pytest.raises(ValueError, match="readback"):
        case.run_case(audit)
    expected = 3 if target == "mass" else 4
    assert len(calls["native"]) == len(audit["readouts"]) == expected
    assert audit["readouts"][-1]["returned"] is True and audit["readouts"][-1]["validated"] is False
    assert audit["readouts"][-1]["paid_return"]["work"]["completed_steps"] == 2


def test_should_preserve_partial_camera_audit_when_main_fails(monkeypatch, tmp_path):
    def run(audit):
        audit["readouts"].append(
            {"target": "camera", "returned": True, "paid_return": {"prediction_returns": 2}}
        )
        raise RuntimeError("synthetic camera summary failure")

    monkeypatch.setattr(case, "STAGE", tmp_path)
    monkeypatch.setattr(case, "preflight", lambda: None)
    monkeypatch.setattr(case, "version", lambda name: "1.26.4")
    monkeypatch.setattr(case.prior, "peak_rss_bytes", lambda: 1)
    monkeypatch.setenv("VK_ICD_FILENAMES", "stub-software-vulkan")
    monkeypatch.setattr(case, "run_case", run)
    assert case.main() == 1
    raw = json.loads((tmp_path / "camera-native-result.json").read_text(encoding="utf8"))
    assert raw["complete"] is False and raw["readouts"][0]["returned"] is True
    assert "synthetic camera summary failure" in raw["error"]
