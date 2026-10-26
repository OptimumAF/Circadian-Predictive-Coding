"""Check all released198 TRAIN inputs without fitting or running a policy.

Read one accepted study and the existing qualified original-correction loader.
Compose the unchanged bounded consistency kernel; save every intended pair and
the actual repeated work. This is a supervision diagnosis, not task evaluation,
new label authority, a controller, or evidence of successful adaptation.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import monotonic
from typing import Any

from scripts.diagnose_learned_robot_control import _episode, _same_push_receipt
from scripts.push_cube_correction_inputs import load_push_correction_inputs
from src.core.cube_taskspace_state import cube_taskspace_spec
from src.core.grouped_input_target_consistency import diagnose_grouped_input_targets

SOURCE = Path("artifacts/runs/aggregated-correction-study-20261021/run.json")
SOURCE_SHA256 = "4563c77a2aeee29c2d0b7c68e077837fc249c47227c4b1a40b068fea3c54af68"
GROUPS = ("current_corrections", "corrections", "push", "a", "b", "train20")
COUNTS = (50, 50, 18, 15, 50, 15)


def _bound_bytes(path: Path, *, work: dict[str, int] | None = None) -> bytes:
    if work is not None:
        work["attempts"] += 1
    with path.open("rb") as handle:
        raw = handle.read(6 * 1024**2 + 1)
    if work is not None:
        work["returns"] += 1
        work["bytes"] += len(raw)
    if len(raw) > 6 * 1024**2 or hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("accepted aggregated study byte binding differs")
    return raw


def read_aggregated_inputs(path: Path = SOURCE):
    """Reuse accepted provenance; metadata alone is never decoded as supervision."""
    read_work = dict(attempts=0, returns=0, bytes=0)
    raw = _bound_bytes(path, work=read_work)
    study = json.loads(raw)
    continuation = study["aggregated_continuation"]
    if (
        study["protocol"] != "push_cube_aggregated_correction_train_v1"
        or study["complete"] is not True
        or study["error"] is not None
        or study["source_unchanged"] is not True
        or study["unreturned_stage_work_unknown"] is not False
        or continuation["complete"] is not True
        or continuation["source_exact"] is not True
        or continuation["actor_exact"] is not True
        or continuation["unreturned_training_work_unknown"] is not False
        or any(
            type(value) is not int
            for value in (
                continuation["rows"],
                *continuation["initial_counts"].values(),
                *continuation["final_counts"].values(),
            )
        )
        or continuation["rows"] != 198
        or continuation["initial_counts"] != dict(updates=120, training_rows=11016)
        or continuation["final_counts"] != dict(updates=128, training_rows=12600)
    ):
        raise ValueError("accepted aggregated TRAIN scope is incomplete or differs")
    original, original_receipt = load_push_correction_inputs()
    if not _same_push_receipt(original_receipt, study["correction_fit_protocol"]["input_receipt"]):
        raise ValueError("original correction provenance differs from accepted study")
    episodes = (
        _episode(study["current_correction_packet"]),
        original,
        _episode(study["training"]),
        *(_episode(value) for value in study["references"]),
    )
    recipe = study["aggregated_correction_protocol"]
    if (
        len(episodes) != 6
        or tuple(len(e.transitions) for e in episodes) != COUNTS
        or any(e.role != "train" or e.spec != cube_taskspace_spec() for e in episodes)
        or len({e.episode_id for e in episodes}) != 6
        or episodes[0].episode_id != "push-cube-correction-targets-17-source120"
        or episodes[1].episode_id != "push-cube-correction-targets-17"
        or recipe["group_order"] != list(GROUPS)
        or recipe["groups"]
        != [
            dict(group=name, episode_id=e.episode_id, rows=rows)
            for name, e, rows in zip(GROUPS, episodes, COUNTS, strict=True)
        ]
    ):
        raise ValueError("complete six-group TRAIN packets or source identities differ")
    metadata: list[dict[str, Any]] = []
    bindings: list[dict[str, Any]] = []
    boundaries: list[list[Any]] = []
    offset = 0
    for index, episode in enumerate(episodes):
        group = "training" if index == 0 else f"reference_{index - 1}"
        identity = dict(
            group=group, episode_id=episode.episode_id, context=episode.context, role="train"
        )
        metadata.append(dict(**identity, rows=len(episode.transitions)))
        bindings.extend(dict(**identity, step=row.step) for row in episode.transitions)
        boundaries.append([group, offset, offset + len(episode.transitions)])
        offset += len(episode.transitions)
    if (
        continuation["original_inputs"] != metadata
        or continuation["row_bindings"] != bindings
        or continuation["groups"] != boundaries
    ):
        raise ValueError("accepted continuation input rows or group boundaries differ")
    return tuple(zip(GROUPS, episodes, strict=True)), dict(
        accepted_study=dict(path=str(path), bytes=len(raw), sha256=SOURCE_SHA256),
        original_correction=original_receipt,
        accepted_study_read_work=read_work,
        scope="accepted released TRAIN packets; no renewed authority or relabeling",
    )


def _require_original_unchanged(receipt, work) -> None:
    # Why: the loader's readback precedes arithmetic; publication must also bind
    # the original packet file after the diagnosis has consumed its owned copy.
    work["attempts"] += 1
    with Path(receipt["path"]).open("rb") as handle:
        raw = handle.read(receipt["bytes"] + 1)
    work["returns"] += 1
    work["bytes"] += len(raw)
    if len(raw) != receipt["bytes"] or hashlib.sha256(raw).hexdigest() != receipt["sha256"]:
        raise ValueError("original correction bytes changed before diagnosis publication")


def run_diagnosis(output: Path, *, source: Path = SOURCE) -> dict[str, Any]:
    """Execute once; retain partial kernel charges and refuse changed source bytes."""
    if not output.parent.is_dir() or output.exists() or output.is_symlink():
        raise ValueError("diagnosis requires a fresh output in an existing directory")
    started = monotonic()
    groups, receipt = read_aggregated_inputs(source)
    diagnosis = diagnose_grouped_input_targets(groups)
    original_final_read = dict(attempts=0, returns=0, bytes=0)
    report: dict[str, Any] = dict(
        definition="aggregated_train_supervision_consistency_v1",
        complete=diagnosis["complete"] is True,
        error=diagnosis.get("error"),
        sources=receipt,
        inputs=[dict(group=name, episode=asdict(episode)) for name, episode in groups],
        diagnosis=diagnosis,
        source_bytes_unchanged=False,
        work=dict(
            model_initializations=0,
            random_draws=0,
            optimizer_updates=0,
            predictions=0,
            gradients=0,
            simulator_controls=0,
            teacher_queries=0,
            accepted_study_reads=receipt["accepted_study_read_work"],
            source_derived_original_correction_reads=2,
            source_derived_original_correction_bytes=2 * receipt["original_correction"]["bytes"],
            original_correction_final_read=original_final_read,
            scope="kernel costs are in diagnosis.work; parsing/copies/validation/Python additional",
        ),
    )
    try:
        _bound_bytes(source, work=receipt["accepted_study_read_work"])
        _require_original_unchanged(receipt["original_correction"], original_final_read)
        report["source_bytes_unchanged"] = True
    except Exception as error:
        report["complete"] = False
        report["source_error"] = f"{type(error).__name__}: {error}"
        report["error"] = report["source_error"]
    report["elapsed_seconds"] = monotonic() - started
    encoded = json.dumps(report, sort_keys=True, allow_nan=False)
    if len(encoded.encode("utf8")) > 16 * 1024**2:
        raise ValueError("grouped diagnosis exceeds16MiB")
    with output.open("x", encoding="utf8") as handle:
        handle.write(encoded + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", type=Path, default=SOURCE)
    args = parser.parse_args()
    result = run_diagnosis(args.output, source=args.source)
    print(json.dumps(dict(complete=result["complete"], error=result["error"])))
    raise SystemExit(0 if result["complete"] else 1)


if __name__ == "__main__":
    main()
