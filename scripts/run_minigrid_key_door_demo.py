"""Collect fixed actual MiniGrid cue/key/door demonstrations; no model updates."""

import argparse
from dataclasses import asdict
import json
import os
from time import perf_counter
from typing import Any

from src.core.demonstration_episode import (
    EpisodeReplaySchema,
    encode_episode,
    decode_episode,
    episode_arrays,
)
from src.core.minigrid_key_door import CueKeyDoorTeacher, predefined_layouts
from src.app.coupled_attribution import AttributionBatch, _validate_batch
from src.infra.minigrid_key_door import MiniGridKeyDoor


def describe_protocol() -> dict[str, Any]:
    return dict(
        protocol_id="minigrid_cue_key_door_demonstrations_v1",
        classification="actual_game_scripted_public_teacher_not_learned_adaptation",
        seed=17,
        max_steps=16,
        observation_dim=148,
        num_classes=7,
        train_sequence=["A", "B", "C_train", "A"],
        final_layout="D",
        cues=["blue", "red"],
        layouts=[asdict(layout) for layout in predefined_layouts()],
        collection_order="all eight TRAIN episodes before two separate final_test demonstrations",
        teacher_inputs="current normalized public observation only; fresh cue/phase memory per episode",
        max_resets=10,
        max_environment_steps=160,
        model_updates=0,
        limitations="scripted success is not learned skill, retention, recovery, generalization or CPC advantage",
    )


def run_demo() -> dict[str, Any]:
    start = perf_counter()
    report: dict[str, Any] = dict(
        status="running", protocol=describe_protocol(), episodes=[], model_updates=0
    )
    layouts = predefined_layouts()
    schedule = [(layout, "train") for layout in (layouts[0], layouts[1], layouts[2], layouts[0])]
    schedule.append((layouts[3], "final_test"))
    seen_rows: set[str] = set()
    for phase, (layout, role) in enumerate(schedule):
        for cue in ("blue", "red"):
            environment = None
            item: dict[str, Any] = dict(
                phase=phase,
                role=role,
                context=layout.context,
                cue=cue,
                status="running",
                teacher_decisions=0,
            )
            report["episodes"].append(item)
            try:
                with MiniGridKeyDoor(
                    layout,
                    cue,
                    seed=17,
                    role=role,
                    episode_id=f"minigrid-{phase}-{layout.context}-{cue}",
                ) as environment:
                    teacher = CueKeyDoorTeacher()
                    for _ in range(16):
                        action = teacher.choose_action(environment.observation)
                        item["teacher_decisions"] += 1
                        environment.step(action)
                        if environment.terminal:
                            break
                    episode = environment.to_demonstration()
                    features, targets = episode_arrays(episode)
                    batch = AttributionBatch(
                        role,
                        layout.context,
                        tuple(
                            f"{episode.episode_id}:step:{row.step}" for row in episode.transitions
                        ),
                        features,
                        targets,
                        episode,
                        7,
                    )
                    _validate_batch(batch, role, seen_rows)
                    item["packet"] = dict(
                        role=role,
                        rows=len(features),
                        feature_shape=features.shape,
                        target_shape=targets.shape,
                        arrays_bytes=features.nbytes + targets.nbytes,
                    )
                    schema = EpisodeReplaySchema(148, len(episode.transitions), 7)
                    if role == "train":
                        encoded, dummy = encode_episode(episode, schema)
                        if decode_episode(encoded, dummy, schema) != episode:
                            raise ValueError("complete training episode codec differed")
                        item["packet"]["replay_array_bytes"] = encoded.nbytes + dummy.nbytes
                    else:
                        try:
                            encode_episode(episode, schema)
                        except ValueError:
                            item["packet"]["replay_refused"] = True
                        else:
                            raise ValueError(
                                "final demonstration was incorrectly admitted to replay"
                            )
                    if environment.report["total_return"] <= 0:
                        raise ValueError("scripted teacher failed the fixed cue/key/door task")
                    item["status"] = "complete"
            except Exception as error:
                item.update(status="failed", reason=f"{type(error).__name__}: {error}")
            finally:
                item["environment"] = None if environment is None else environment.report
    report["status"] = (
        "complete" if all(row["status"] == "complete" for row in report["episodes"]) else "failed"
    )
    report["environment_steps"] = sum(
        row["environment"]["environment_steps"]
        for row in report["episodes"]
        if row["environment"] is not None
    )
    report["seconds"] = perf_counter() - start
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--describe", action="store_true", help="Preview without constructing environments"
    )
    args = parser.parse_args()
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    report = describe_protocol() if args.describe else run_demo()
    print(json.dumps(report, allow_nan=False, indent=2))
    if not args.describe and report["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
