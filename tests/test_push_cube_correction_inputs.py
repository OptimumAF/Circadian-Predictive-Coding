"""Synthetic byte/state/history/packet qualification; no model, teacher or SDK."""

from dataclasses import asdict
import hashlib
import json

import pytest

from scripts import push_cube_correction_inputs as intake
from src.core.continuous_demonstration import ContinuousEpisode, ContinuousTransition
from src.core.cube_taskspace_state import (
    CubeTaskSpaceState,
    cube_taskspace_spec,
    cube_taskspace_observation,
)
from src.core.pick_cube_action_history import ActionHistoryState, advance_action_history
from src.core.push_cube_corrections import (
    PROVENANCE,
    PushCubeCorrectionBatch,
    PushCubeCorrectionLabel,
)
from src.infra.maniskill_push_cube import PushCubeWork


def synthetic_release():
    state = CubeTaskSpaceState(False, (0.0, 0.0, 0.12), (0.0, 0.0, 0.02), (0.0, 0.0, 0.02))
    history = ActionHistoryState()
    rows = []
    labels = []
    for step in range(50):
        observation = cube_taskspace_observation(state, history)
        action = (0.0, 0.0, 0.0, 1.0)
        history = advance_action_history(history, action)
        rows.append(
            ContinuousTransition(
                step,
                observation,
                action,
                0.0,
                cube_taskspace_observation(state, history),
                False,
                step == 49,
            )
        )
        labels.append(
            PushCubeCorrectionLabel(step, state, observation, (0.0, 0.0, 0.0, -1.0), "hold")
        )
    episode = ContinuousEpisode(
        "push-cube-student-17", "push-cube-train-17", "train", cube_taskspace_spec(), tuple(rows)
    )
    batch = PushCubeCorrectionBatch(17, episode, tuple(labels))
    report = dict(
        definition=PROVENANCE,
        complete=True,
        source_unchanged=True,
        training_release=True,
        seed=17,
        role="train",
        source_updates=112,
        source_training_rows=9832,
        episode=asdict(episode),
        labels=[asdict(x) for x in labels],
        states=[asdict(state)] * 50,
        transitions=[asdict(row) for row in rows],
        label_receipts=[
            dict(step=x.step, observed_at=x.observed_at, labels_arrived_at=x.labels_arrived_at)
            for x in labels
        ],
        work=asdict(PushCubeWork(1, 1, 1, 1, 50, 50, 250, 0, 51, 51, 51, 51, 102, 102, 1, 1)),
        unreturned_query_possible=False,
        unreturned_prediction_possible=False,
        unreturned_native_work_unknown=False,
    )
    for key in (
        "teacher_query_attempts",
        "teacher_query_returns",
        "prediction_attempts",
        "prediction_returns",
        "control_attempts",
        "control_returns",
    ):
        report[key] = 50
    return dict(
        complete=True,
        source_unchanged=True,
        unreturned_stage_work_unknown=False,
        push_corrections=report,
        push_correction_target_packet=asdict(batch.training_episode(now=1100)),
        push_correction_release_at=1100,
    )


@pytest.mark.parametrize(
    "fault", [None, "hash", "oversize", "role", "counter", "work", "history", "packet"]
)
def test_should_admit_only_original_complete_owned_train_correction_packet(
    tmp_path, monkeypatch, fault
):
    raw = synthetic_release()
    if fault == "role":
        raw["push_corrections"]["episode"]["role"] = "final_test"
    if fault == "counter":
        raw["push_corrections"]["source_updates"] = 112.0
    if fault == "work":
        raw["push_corrections"]["work"]["completed_steps"] = 49
    if fault == "history":
        raw["push_corrections"]["labels"][1]["observation"] = (0.0,) * 12
    if fault == "packet":
        raw["push_correction_target_packet"]["transitions"][0]["action"] = (0.1, 0.0, 0.0, -1.0)
    payload = (
        b" " * (4 * 1024**2 + 1)
        if fault == "oversize"
        else json.dumps(raw, allow_nan=False).encode()
    )
    path = tmp_path / "synthetic-correction.json"
    path.write_bytes(payload)
    monkeypatch.setattr(
        intake,
        "CORRECTION_SHA256",
        "0" * 64 if fault == "hash" else hashlib.sha256(payload).hexdigest(),
    )
    if fault is not None:
        with pytest.raises(ValueError):
            intake.load_push_correction_inputs(path)
    else:
        packet, receipt = intake.load_push_correction_inputs(path)
        assert json.dumps(asdict(packet), sort_keys=True) == json.dumps(
            raw["push_correction_target_packet"], sort_keys=True
        )
        assert (receipt["rows"], receipt["release_at"]) == (50, 1100)
        assert (
            packet.transitions[1].observation[7:11] == (0.0, 0.0, 0.0, 1.0)
            and packet.transitions[1].action[3] == -1.0
        )
    assert path.read_bytes() == payload
