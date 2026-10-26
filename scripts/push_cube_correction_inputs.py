"""Admit accepted provisional Push packets without collecting or relabeling.

Own byte/semantic binding and reuse core physical-state/history/arrival validation.
No model, teacher, simulator, fitting, row selection or final/probe access.
"""

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

from scripts.joint_acquisition_inputs import _episode
from src.core.cube_taskspace_state import CubeTaskSpaceState
from src.core.push_cube_corrections import (
    PROVENANCE,
    PushCubeCorrectionBatch,
    PushCubeCorrectionLabel,
)
from src.infra.maniskill_push_cube import PushCubeWork

CORRECTION_INPUT = Path("artifacts/runs/push-on-policy-corrections-20261021/run.json")
CORRECTION_SHA256 = "1a4023a0f75bec3d338ab32c2e4e3ff5a8a9c79206689aac0972121b82544a78"


BALANCED_CORRECTION_INPUT = Path("artifacts/runs/balanced-student-corrections-20261021/run.json")
BALANCED_CORRECTION_SHA256 = "0d0e94896e9b62ca24f85ebb634f54b86643f230100f3e02bcbde263376895c7"


def _json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def load_push_correction_inputs(path: Path = CORRECTION_INPUT):
    """Return the entire released50-row TRAIN112 packet and its original receipt."""
    return _load_push_correction_inputs(
        path, expected_sha256=CORRECTION_SHA256, source_updates=112, source_training_rows=9832
    )


def load_balanced_push_correction_inputs(path: Path = BALANCED_CORRECTION_INPUT):
    """Admit observed TRAIN120 labels with a distinct derived target-packet ID.

    Why: both batches share the observed student context/episode IDs. A derived
    target ID distinguishes training inputs without changing original data,
    label clocks, consent, receipts, or source artifact provenance.
    """
    packet, receipt = _load_push_correction_inputs(
        path,
        expected_sha256=BALANCED_CORRECTION_SHA256,
        source_updates=120,
        source_training_rows=11016,
        protocol="push_cube_balanced_student_corrections_train_v1",
    )
    derived = replace(packet, episode_id="push-cube-correction-targets-17-source120")
    return derived, dict(
        receipt, original_episode_id=packet.episode_id, derived_episode_id=derived.episode_id
    )


def _load_push_correction_inputs(
    path: Path,
    *,
    expected_sha256: str,
    source_updates: int,
    source_training_rows: int,
    protocol: str | None = None,
):
    """Share exact byte/core/physical-history qualification across the two releases."""
    with path.open("rb") as handle:
        payload = handle.read(4 * 1024**2 + 1)
    if len(payload) > 4 * 1024**2 or hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("accepted Push correction binding differs")
    raw = json.loads(payload)
    report = raw["push_corrections"]
    if protocol is not None:
        header = raw.get("push_correction_protocol")
        if (
            raw.get("protocol") != protocol
            or type(header) is not dict
            or header.get("checkpoint") != source_updates
            or type(header.get("checkpoint")) is not int
            or header.get("source_training_rows") != source_training_rows
            or type(header.get("source_training_rows")) is not int
            or header.get("seed") != 17
            or type(header.get("seed")) is not int
            or header.get("role") != "train"
        ):
            raise ValueError("accepted balanced Push correction protocol differs")
    count_keys = (
        "teacher_query_attempts",
        "teacher_query_returns",
        "prediction_attempts",
        "prediction_returns",
        "control_attempts",
        "control_returns",
    )
    if (
        raw["complete"] is not True
        or raw["source_unchanged"] is not True
        or raw["unreturned_stage_work_unknown"] is not False
        or report["definition"] != PROVENANCE
        or report["complete"] is not True
        or report["training_release"] is not True
        or report["source_unchanged"] is not True
        or report["seed"] != 17
        or type(report["seed"]) is not int
        or report["role"] != "train"
        or report["source_updates"] != source_updates
        or report["source_training_rows"] != source_training_rows
        or type(report["source_updates"]) is not int
        or type(report["source_training_rows"]) is not int
        or any(type(report[key]) is not int or report[key] != 50 for key in count_keys)
        or any(
            report[key] is not False
            for key in (
                "unreturned_query_possible",
                "unreturned_prediction_possible",
                "unreturned_native_work_unknown",
            )
        )
        or len(report["labels"]) != 50
        or len(report["states"]) != 50
        or raw["push_correction_release_at"] != 1100
        or type(raw["push_correction_release_at"]) is not int
    ):
        raise ValueError("accepted Push correction release is incomplete")
    expected_work = asdict(PushCubeWork(1, 1, 1, 1, 50, 50, 250, 0, 51, 51, 51, 51, 102, 102, 1, 1))
    if _json(report["work"]) != _json(expected_work):
        raise ValueError("accepted Push correction paid work differs")
    labels = tuple(
        PushCubeCorrectionLabel(
            value["step"],
            CubeTaskSpaceState(
                value["state"]["is_grasped"],
                tuple(value["state"]["tcp_position"]),
                tuple(value["state"]["object_position"]),
                tuple(value["state"]["goal_position"]),
            ),
            tuple(value["observation"]),
            tuple(value["target"]),
            value["stage"],
        )
        for value in report["labels"]
    )
    batch = PushCubeCorrectionBatch(17, _episode(report["episode"]), labels)
    packet = batch.training_episode(now=1100)
    receipts = [
        dict(
            step=label.step,
            observed_at=label.observed_at,
            labels_arrived_at=label.labels_arrived_at,
        )
        for label in batch.labels
    ]
    # Why: core copying must not hide coercion, changed physical inputs or targets.
    expected = dict(
        episode=report["episode"],
        labels=report["labels"],
        states=report["states"],
        transitions=report["transitions"],
        receipts=report["label_receipts"],
        packet=raw["push_correction_target_packet"],
    )
    actual = dict(
        episode=asdict(batch.episode),
        labels=[asdict(label) for label in batch.labels],
        states=[asdict(label.state) for label in batch.labels],
        transitions=[asdict(row) for row in batch.episode.transitions],
        receipts=receipts,
        packet=asdict(packet),
    )
    if _json(actual) != _json(expected):
        raise ValueError("Push correction decoded evidence or released target packet differs")
    with path.open("rb") as handle:
        current = handle.read(4 * 1024**2 + 1)
    if current != payload:
        raise ValueError("Push correction input changed during qualification")
    return packet, dict(
        path=str(path),
        sha256=expected_sha256,
        bytes=len(payload),
        rows=50,
        source_updates=source_updates,
        source_training_rows=source_training_rows,
        release_at=1100,
        original_label_receipts=receipts,
    )
