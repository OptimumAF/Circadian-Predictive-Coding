"""Fixed MiniGrid A-return data, report validation and cost descriptions.

Inputs are original A TRAIN monitors and the existing probe's nested return DTO.
Outputs are two owned update templates and plain bounded summaries. No learner,
physics, final data, task selection, promotion or budget renewal authority.
"""

from dataclasses import replace
from typing import Any

from src.app.coupled_attribution import AttributionBatch, _copy_batch, _validate_batch


def return_training_sources(monitors: tuple[AttributionBatch, ...]) -> tuple[AttributionBatch, ...]:
    if type(monitors) is not tuple or len(monitors) != 2:
        raise ValueError("A return requires exactly two original TRAIN monitors")
    seen: set[str] = set()
    for packet, cue in zip(monitors, ("blue", "red"), strict=True):
        _validate_batch(packet, "train", seen)
        if (
            packet.episode is None
            or packet.context != "A"
            or packet.num_classes != 7
            or packet.features.shape not in ((9, 148), (9, 296))
            or not packet.episode.episode_id.endswith(cue)
        ):
            raise ValueError("A return requires original blue/red A TRAIN episodes")
    # Why two templates: retaining16 additional history packets would exceed the
    # existing combined65536-entry ceiling. The fixed schedule reuses owned data.
    result = []
    for packet, cue in zip(monitors, ("blue", "red"), strict=True):
        assert packet.episode is not None
        result.append(
            _copy_batch(
                replace(
                    packet,
                    episode=replace(packet.episode, episode_id=f"minigrid-return-train-{cue}"),
                    row_ids=tuple(f"minigrid-return-train-{cue}:step:{i}" for i in range(9)),
                )
            )
        )
    return tuple(result)


def describe_return(protocol: dict[str, Any]) -> None:
    methods = protocol["methods"]
    protocol["protocol_id"] = protocol["protocol_id"].removesuffix("_v1") + "_a_return_v1"
    protocol["return_phase"] = dict(
        context="A",
        role="train",
        updates=16,
        schedule=[0, 1] * 8,
        checkpoint_updates=[0, 4, 8, 16],
        input_array_bytes_by_method={m: 21456 if m == "recurrent_bc" else 42768 for m in methods},
        scope="same disposable learner/optimizer after F16; separately admitted A TRAIN update templates and fresh independent budget; monitors remain readonly",
        evaluation="F probe_test and original A TRAIN monitors at return0/4/8/16; return0 explicitly re-predicts and must equal preceding F16/A16",
        direction="CE gain=return0-return16; accuracy/exact gain=return16-return0; fit recovery/maintenance only, not acquired-task retention, live success or generalization",
        storage="two additional owned update templates; combined F+monitor+return470448B/history or236016B/recurrent per probe; unchanged65536-entry ceiling; copies/reports/native state/RSS additional",
    )
    work = protocol["planned_work"]
    work.update(
        return_native_train_calls=32 * len(methods),
        return_training_rows=288 * len(methods),
        return_score_forwards=32 * len(methods),
        return_score_rows=288 * len(methods),
    )
    for key, extra in (
        ("native_train_calls", work["return_native_train_calls"]),
        ("training_rows", work["return_training_rows"]),
        ("score_forwards", work["return_score_forwards"]),
        ("score_rows", work["return_score_rows"]),
    ):
        work[key] += extra
    if "recurrent_bc" in methods:
        work["recurrent_unroll_steps"] += 576


def validate_return_report(report, *, input_bytes, monitor_keys, prior_report) -> None:
    if type(report) is not dict or report.get("status") != "complete":
        raise ValueError("A return failed or did not run")
    for key, expected in dict(
        completed_updates=16,
        attempted_updates=16,
        completed_training_examples=144,
        attempted_training_examples=144,
        score_forward_calls_completed=8,
        score_forward_calls_attempted=8,
        reserved_data_bytes=input_bytes,
    ).items():
        if type(report.get(key)) is not int or report[key] != expected:
            raise ValueError("A return counts/storage differ from the fixed recipe")
    if [point["updates"] for point in report["checkpoints"]] != [0, 4, 8, 16] or any(
        type(p["updates"]) is not int or set(p["scores"]) != {"F-blue", "F-red"}
        for p in report["checkpoints"]
    ):
        raise ValueError("A return F checkpoints differ")
    old = report.get("training_readouts")
    if type(old) is not dict or old.get("status") != "complete":
        raise ValueError("A return monitor did not complete")
    for key, expected in dict(
        score_forward_calls_completed=8,
        score_forward_calls_attempted=8,
        completed_rows=72,
        attempted_rows=72,
        reserved_data_bytes=0,
    ).items():
        if type(old.get(key)) is not int or old[key] != expected:
            raise ValueError("A return monitor counts/storage differ")
    if [p["updates"] for p in old["checkpoints"]] != [0, 4, 8, 16] or any(
        type(p["updates"]) is not int
        or p.get("status") != "complete"
        or set(p["scores"]) != monitor_keys
        or set(p.get("decision_diagnostics", {})) != monitor_keys
        for p in old["checkpoints"]
    ):
        raise ValueError("A return A checkpoints differ")
    for current, previous in (
        (report["checkpoints"][0], prior_report["checkpoints"][-1]),
        (old["checkpoints"][0], prior_report["training_readouts"]["checkpoints"][-1]),
    ):
        if any(current[key] != previous[key] for key in ("scores", "decision_diagnostics")):
            raise ValueError("A return initial F/A readouts differ from preceding F16")


def sum_return_work(phases: list[dict[str, Any]]) -> dict[str, Any]:
    monitored = [v["training_readouts"] for v in phases if "training_readouts" in v]
    return dict(
        native_train_calls=sum(v["completed_updates"] for v in phases),
        training_rows=sum(v["completed_training_examples"] for v in phases),
        score_forwards_attempted=sum(v["score_forward_calls_attempted"] for v in phases)
        + sum(v["score_forward_calls_attempted"] for v in monitored),
        score_forwards_completed=sum(v["score_forward_calls_completed"] for v in phases)
        + sum(v["score_forward_calls_completed"] for v in monitored),
        score_rows_completed=9 * sum(v["score_forward_calls_completed"] for v in phases)
        + sum(v["completed_rows"] for v in monitored),
        reserved_data_bytes=sum(v.get("reserved_data_bytes") or 0 for v in phases),
        storage_may_be_unobserved=any(v.get("reserved_data_bytes") is None for v in phases),
        elapsed_seconds=sum(v.get("elapsed_seconds") or 0 for v in phases),
        failure_work_may_be_unobserved=any(
            v["attempted_updates"] > v["completed_updates"]
            or v["score_forward_calls_attempted"] > v["score_forward_calls_completed"]
            for v in phases
        )
        or any(v["failure_work_may_be_unobserved"] for v in monitored),
        scope="separate fixed A-return updates and F/A readouts on disposable learner; returned work only; original monitor arrays reused with0new input-array bytes; incomplete timing/copies/Python/native state/RSS additional",
    )
