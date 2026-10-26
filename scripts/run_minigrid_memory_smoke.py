"""Run a fixed, bounded actual-environment smoke; no learning or model selection."""

import argparse
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
from src.infra.minigrid_memory import MiniGridMemory, ENVIRONMENT_ID, ENCODING_ID, VERSIONS

UPPER_BRANCH_ACTIONS = (2, 2, 2, 2, 2, 0, 2)


def describe_protocol() -> dict[str, Any]:
    return dict(
        protocol_id="minigrid_memory_boundary_smoke_v1",
        classification="scripted_engineering_trace_not_expert_learning_or_comparison",
        environment_id=ENVIRONMENT_ID,
        versions=dict(VERSIONS),
        encoding_id=ENCODING_ID,
        seed=17,
        max_steps=16,
        role="train",
        observation_dim=148,
        num_classes=7,
        actions=[list(UPPER_BRANCH_ACTIONS), [0] * 16, list(UPPER_BRANCH_ACTIONS)],
        max_environment_resets=3,
        max_environment_steps=30,
        model_updates=0,
        physics_accounting="one native step per action, including no-ops; no retries",
        mission="fixed Memory instruction retained in raw evidence, no target reads",
        limitations="not key/door, A-B-C-A, heldout, reward learning, useful retention or circadian benefit",
    )


def run_smoke() -> dict[str, Any]:
    start = perf_counter()
    reports = []
    for actions in (UPPER_BRANCH_ACTIONS, (0,) * 16, UPPER_BRANCH_ACTIONS):
        with MiniGridMemory(17, episode_id="memory-seed17") as environment:
            for action in actions:
                environment.step(action)
                if environment.terminal:
                    break
            report = environment.report
            if not environment.terminal:
                raise ValueError("frozen script did not complete its bounded episode")
            if report["terminated"] and not report["truncated"]:
                episode = environment.to_demonstration()
                features, targets = episode_arrays(episode)
                schema = EpisodeReplaySchema(148, len(episode.transitions), 7)
                encoded, dummy = encode_episode(episode, schema)
                if decode_episode(encoded, dummy, schema) != episode:
                    raise ValueError("existing episode replay roundtrip differed")
                report["demonstration_geometry"] = dict(
                    role=episode.role,
                    rows=len(features),
                    feature_shape=features.shape,
                    target_shape=targets.shape,
                    array_bytes=features.nbytes + targets.nbytes,
                    replay_array_bytes=encoded.nbytes + dummy.nbytes,
                    expert_claim=False,
                )
        report["closed"] = environment.report["closed"]
        reports.append(report)
    if reports[0] != reports[2]:
        raise ValueError("same frozen seed and action script produced different evidence")
    if not reports[1]["truncated"] or reports[1]["terminated"]:
        raise ValueError("fixed turning script must retain native timeout flags")
    return dict(
        status="complete",
        protocol=describe_protocol(),
        episodes=reports,
        deterministic_repeat_equal=True,
        environment_resets=3,
        environment_steps=sum(row["environment_steps"] for row in reports),
        model_updates=0,
        seconds=perf_counter() - start,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--describe",
        action="store_true",
        help="Print protocol without loading optional dependencies",
    )
    args = parser.parse_args()
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    print(
        json.dumps(describe_protocol() if args.describe else run_smoke(), allow_nan=False, indent=2)
    )


if __name__ == "__main__":
    main()
