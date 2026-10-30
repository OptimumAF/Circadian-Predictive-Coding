"""Real calibrator/observer on literal probabilities; no fitted model or teacher."""

from copy import deepcopy
from dataclasses import dataclass, replace

import numpy as np
import pytest

from scripts.arrived_candidate_shift import ArrivedCandidateShift
from src.app.coupled_attribution import AttributionBatch
from src.core.reference_controller import ReferenceObservation, ReferencePolicy, choose_reference
from src.core.shift_calibration import ShiftCalibration


@dataclass
class State:
    completed_updates: int = 12


@dataclass
class Snapshot:
    actor_version: str
    learner_version: str
    wake_updates: int
    state: State


class Runtime:
    def __init__(self):
        self.completed = 0
        self.version = "source"
        self.learner = "c0"
        self.native = 12
        self.snapshot_calls = 0

    def candidate_snapshot(self):
        self.snapshot_calls += 1
        return Snapshot(
            actor_version=self.version,
            learner_version=self.learner,
            wake_updates=self.completed,
            state=State(self.native),
        )


def batches(values):
    return tuple(
        AttributionBatch(
            "train",
            f"batch-{i}",
            (f"row-{i}",),
            np.array([[float(v)]]),
            np.array([[0.0]]),
            num_classes=6,
        )
        for i, v in enumerate(values)
    )


def setup(*, prefix=5, margin=1e-6):
    calls = []

    def literal_prediction(state, features):
        calls.append((deepcopy(state), features.copy()))
        result = np.zeros((len(features), 6))
        result[:, 0] = 1.0 if state.completed_updates == 12 else 0.25
        result[:, 1] = 1 - result[:, 0]
        return result

    rows = tuple((f"row-{i}",) for i in range(prefix))
    reader = ArrivedCandidateShift(
        literal_prediction,
        calibration=ShiftCalibration(prefix, margin),
        calibration_model_id="declared-reference",
        calibration_prefix_rows=rows,
    )
    return reader, Runtime(), calls


def read(reader, runtime, data, index):
    return reader.read(
        runtime,
        data,
        arrived=index + 1,
        completed=runtime.completed,
        tick=index + 1,
        label_delay=1,
        actor_version=runtime.version,
        require_input=lambda: None,
    )


@pytest.mark.parametrize(
    "change",
    [
        dict(calibration=None),
        dict(calibration=object()),
        dict(calibration_model_id=None),
        dict(calibration_model_id=""),
        dict(calibration_model_id="x" * 129),
        dict(calibration_prefix_rows=None),
        dict(calibration_prefix_rows=[]),
        dict(calibration_prefix_rows=(("row-0",),) * 5),
        dict(calibration_prefix_rows=tuple(("x" * 129 + str(i),) for i in range(5))),
        dict(calibration_prefix_rows=((),) * 5),
        dict(calibration=ShiftCalibration(8)),
        dict(
            calibration=ShiftCalibration(4),
            calibration_prefix_rows=tuple((f"row-{i}",) for i in range(4)),
        ),
        dict(
            calibration_prefix_rows=(
                tuple(f"many-{i}" for i in range(44)),
                ("b",),
                ("c",),
                ("d",),
                ("e",),
            )
        ),
    ],
)
def test_should_refuse_incomplete_invalid_or_unbounded_declarations_before_prediction(change):
    args = dict(
        calibration=ShiftCalibration(5),
        calibration_model_id="declared",
        calibration_prefix_rows=tuple((f"row-{i}",) for i in range(5)),
    )
    args.update(change)
    with pytest.raises(ValueError):
        ArrivedCandidateShift(lambda *a: pytest.fail("prediction"), **args)


@pytest.mark.parametrize("clear", [False, True])
def test_should_suppress_prefix_then_preserve_onset_continued_or_clear_without_training(clear):
    reader, runtime, calls = setup()
    data = batches((0,) * 5 + (1, 1, -1 if clear else 1))
    feedback = []
    for i in range(8):
        feedback.append(read(reader, runtime, data, i))
    report = reader.report
    assert [row["decision"]["phase"] for row in report["records"]][:5] == ["calibration"] * 5
    assert all(not any(f.alarms) for f in feedback[:6])
    assert feedback[6].alarms == (False, True, False)
    feature = [row["incident"]["states"][1] for row in report["records"]]
    assert feature[6] == "onset" and feature[7] == ("cleared" if clear else "continued")
    result = report["calibration"]["result"]
    assert (
        result["calibration_batches"]
        == result["calibration_rows"]
        == result["replayed_observations"]
        == 5
    )
    assert result["policy"]["feature_threshold"] == 1e-6
    assert len(calls) == report["attempts"] == report["returns"] == 8
    assert report["calibration"]["retained_native_state_copies"] == 2
    assert runtime.completed == 0 and all(state.completed_updates == 12 for state, _ in calls)
    decision = choose_reference(
        ReferencePolicy("drift"), ReferenceObservation(7, 1, 7, 8, "source", feedback[6])
    )
    assert decision.action_id == "wake1"


def test_should_preserve_unreachable_thresholds_and_negative_monitor_result():
    reader, runtime, calls = setup(margin=1.0)
    data = batches((0,) * 5 + (1, 1, 1))
    assert all(not any(read(reader, runtime, data, i).alarms) for i in range(8))
    result = reader.report["calibration"]["result"]
    assert result["reachable"] == (False, False, False)
    assert result["policy"]["feature_threshold"] == 1.0
    assert len(calls) == 8


def test_should_keep_fixed_diagnostic_model_when_live_candidate_learns():
    reader, runtime, calls = setup(prefix=6)
    data = batches((0,) * 8)
    for i in range(6):
        read(reader, runtime, data, i)
    runtime.completed, runtime.native = 1, 13
    assert not any(read(reader, runtime, data, 6).alarms)
    row = reader.report["records"][-1]
    assert (row["wake_updates"], row["native_updates"], row["prediction_native_updates"]) == (
        1,
        13,
        12,
    )
    assert calls[-1][0].completed_updates == 12
    assert reader.report["calibration"]["reference_binding"] == ("source", "c0", 0, 12)


def test_should_leave_default_on_current_candidate_and_preserve_report_schema():
    calls = []

    def literal_prediction(state, features):
        calls.append(state.completed_updates)
        return np.tile(np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]), (len(features), 1))

    reader = ArrivedCandidateShift(literal_prediction)
    runtime = Runtime()
    data = batches((0, 0))
    read(reader, runtime, data, 0)
    runtime.completed, runtime.native = 1, 13
    read(reader, runtime, data, 1)
    assert calls == [12, 13]
    assert "calibration" not in reader.report
    assert reader.report["definition"] == "arrived_candidate_categorical_shift_v1"
    assert "prediction_model" not in reader.report["records"][0]


@pytest.mark.parametrize("role", ["probe_train", "probe_test", "final"])
def test_should_refuse_non_train_prefix_before_copy_or_prediction(role):
    reader, runtime, calls = setup()
    data = batches((0,))
    with pytest.raises(ValueError):
        read(reader, runtime, (replace(data[0], role=role),), 0)
    assert not calls and runtime.snapshot_calls == 0


def test_should_refuse_changed_prefix_permanently_and_preserve_prior_charges():
    reader, runtime, calls = setup()
    data = batches((0, 0))
    read(reader, runtime, data, 0)
    changed = replace(data[1], row_ids=("replacement",))
    with pytest.raises(ValueError, match="predeclared"):
        read(reader, runtime, (data[0], changed), 1)
    assert reader.report["attempts"] == reader.report["returns"] == len(calls) == 1
    with pytest.raises(ValueError, match="stopped"):
        read(reader, runtime, data, 1)


@pytest.mark.parametrize(
    "fault",
    [
        "callback",
        "config",
        "same_config",
        "label",
        "prefix",
        "reference",
        "same_reference",
        "detector",
    ],
)
def test_should_refuse_changed_owned_calibration_without_rebasing(fault):
    reader, runtime, calls = setup()
    data = batches((0, 0))
    read(reader, runtime, data, 0)
    if fault == "callback":
        reader._predict = lambda *a: pytest.fail("replaced prediction")
    elif fault == "config":
        object.__setattr__(reader._calibration, "margin", 0.5)
    elif fault == "same_config":
        reader._calibration = ShiftCalibration(5)
    elif fault == "label":
        reader._calibration_model_id = "other"
    elif fault == "prefix":
        reader._prefix_rows = tuple(list(reader._prefix_rows))
    elif fault == "reference":
        reader._reference_state.completed_updates = 99
    elif fault == "same_reference":
        reader._reference_state = deepcopy(reader._reference_state)
    else:
        reader._detector = object()
    with pytest.raises(ValueError, match="original calibration"):
        read(reader, runtime, data, 1)
    assert reader.report["stopped"] and reader.report["attempts"] == len(calls) == 1
    assert reader.report["calibration"]["configuration"]["margin"] == 1e-6


@pytest.mark.parametrize("fault", ["state", "features", "lost"])
def test_should_keep_failure_costs_and_not_accept_mutated_working_reference(fault):
    reader, runtime, calls = setup()
    data = batches((0,))
    original = reader._predict

    def literal_prediction(state, features):
        result = original(state, features)
        if fault == "state":
            state.completed_updates = 99
        elif fault == "features":
            features[:] = 1
        else:
            raise RuntimeError("lost literal return")
        return result

    # Construct with the declared callback, rather than replacing a pinned port.
    reader = ArrivedCandidateShift(
        literal_prediction,
        calibration=ShiftCalibration(5),
        calibration_model_id="declared-reference",
        calibration_prefix_rows=tuple((f"row-{i}",) for i in range(5)),
    )
    with pytest.raises((ValueError, RuntimeError)):
        read(reader, runtime, data, 0)
    assert reader.report["attempts"] == 1
    assert reader.report["returns"] == (0 if fault == "lost" else 1)
    assert not reader.report["records"][0]["complete"] and reader.report["stopped"]
    assert runtime.completed == 0


def test_should_not_open_future_data_or_copy_state_without_new_arrival_and_keep_detached_reports():
    reader, runtime, calls = setup()
    report = reader.report
    report["calibration"]["model_id"] = "tampered"
    with pytest.raises(ValueError, match="cost report"):
        reader.require_paid(report)
    reader.require_paid(reader.report)

    class Closed:
        def __getitem__(self, index):
            pytest.fail("future packet opened")

    assert (
        reader.read(
            runtime,
            Closed(),
            arrived=0,
            completed=0,
            tick=0,
            label_delay=1,
            actor_version="source",
            require_input=lambda: pytest.fail("source opened"),
        )
        is None
    )
    assert not calls and runtime.snapshot_calls == 0


@pytest.mark.parametrize("delegate", ["detector", "incident"])
def test_should_preserve_completed_scalar_fit_when_final_prefix_return_is_lost(
    delegate, monkeypatch
):
    reader, runtime, calls = setup()
    data = batches((0,) * 5)
    for i in range(4):
        read(reader, runtime, data, i)
    target = reader._detector if delegate == "detector" else reader._incidents
    original = target.observe

    def lost(value):
        original(value)
        raise RuntimeError("lost scalar return")

    monkeypatch.setattr(target, "observe", lost)
    with pytest.raises(RuntimeError, match="lost scalar return"):
        read(reader, runtime, data, 4)
    report = reader.report
    assert report["attempts"] == report["returns"] == len(calls) == 5
    assert report["calibration"]["result"]["calibration_batches"] == 5
    assert report["calibration"]["result"]["replayed_observations"] == 5
    assert report["stopped"] and not report["records"][-1]["complete"]
    with pytest.raises(ValueError, match="stopped"):
        read(reader, runtime, data, 4)
