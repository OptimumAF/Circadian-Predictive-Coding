"""Qualify one official seed17 TRAIN expert conversion; never train a learner.

Usage: python -m scripts.maniskill_expert_smoke --output fresh-expert.json
Requires the pinned optional CPU ManiSkill stack and a prospective local budget.
"""

import argparse
from dataclasses import asdict
import faulthandler
from importlib.metadata import PackageNotFoundError, version
import json
import logging
from pathlib import Path
import time

from src.infra.maniskill_expert import collect_pick_cube_expert


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.output.open("x", encoding="utf8") as handle:
        # Why: a native fatal signal bypasses Python finally/result serialization.
        # Flush each boundary marker and retain Python frames in the native log.
        faulthandler.enable()
        logger = logging.getLogger("src.infra.maniskill_expert")
        previous_level, previous_propagation = logger.level, logger.propagate
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        start = time.monotonic()
        try:
            result = collect_pick_cube_expert()
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous_level)
            logger.propagate = previous_propagation
            handler.close()
        payload = asdict(result)
        versions: dict[str, str | None] = {}
        for name in ("mani_skill", "sapien", "gymnasium", "torch", "numpy"):
            try:
                versions[name] = version(name)
            except PackageNotFoundError:
                versions[name] = None
        payload.update(
            qualified=result.qualified,
            seed=17,
            role="train",
            task="PickCube-v1",
            source_controller="pd_joint_pos",
            target_controller="pd_ee_delta_pos",
            optimizer_updates=0,
            reset_physics_steps=None,
            constructor_resets="official seed2022 per scene attempt; progress unknown on failed creation",
            reference_replay="bounded prefix, success observed separately; not full replay identity",
            planning_calls="internal official planner work unmeasured",
            versions=versions,
            elapsed_seconds=time.monotonic() - start,
        )
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "candidate_episode"}))
    return 0 if result.qualified else 1


if __name__ == "__main__":
    raise SystemExit(main())
