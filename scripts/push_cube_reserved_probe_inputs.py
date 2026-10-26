"""Load the fixed qualified reserved PickCube model-space probe pair.

Inputs are the original exported probe_train23/probe_test24 JSON files, bound by
literal digests. Output is an owned one-update diagnostic plan and input receipts.
This helper does not collect, qualify teacher authority, relabel, fit, train,
score, run physics or expose an adapted learner.
"""

import hashlib
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from src.adapters.continuous_probe import ContinuousProbePlan
from src.core.cube_taskspace_state import cube_taskspace_spec
from src.core.pick_cube_action_history import validate_action_history_probe_episode
from src.core.pick_cube_probe_context import pick_cube_probe_binding
from src.core.pick_cube_taskspace_history import PickCubeTaskSpaceHistoryRecipe
from src.infra.continuous_probe_json import (
    MAX_PROBE_FILE_BYTES,
    read_continuous_probe_episode,
)

_RUNS = Path(__file__).resolve().parents[1] / "artifacts" / "runs"
PROBES = {
    "probe_train": (
        _RUNS / "reserved-robot-probe-input-native-20261020/probe_train-model.json",
        "31fd5af02bf6646e9a945fb20b0548571a93378f0144fbe9a0021df9d028fcb0",
    ),
    "probe_test": (
        _RUNS / "reserved-robot-probe-input-native-20261020/probe_test-model.json",
        "cefe95a9fcbb36d13ff7111fc0bdb2650b7d9e5cd4891609924fed069a932194",
    ),
}


def _bytes(path: Path) -> bytes:
    with path.open("rb") as handle:
        payload = handle.read(MAX_PROBE_FILE_BYTES + 1)
    if len(payload) > MAX_PROBE_FILE_BYTES:
        raise ValueError("reserved probe exceeds the existing2MiB input bound")
    return payload


def load_reserved_probe_inputs() -> tuple[ContinuousProbePlan, list[dict[str, Any]]]:
    """Load the fixed split without granting new collection or training authority.

    Why: the existing reader opens its own file. Compare the complete original
    bytes again after parsing, rather than treating a pathname hash as proof of
    which version was read. The reader's original byte receipt is retained.
    """
    recipe = PickCubeTaskSpaceHistoryRecipe(mode="executed")
    episodes = []
    receipts: list[dict[str, Any]] = []
    for role in ("probe_train", "probe_test"):
        path, expected_digest = PROBES[role]
        original = _bytes(path)
        digest = hashlib.sha256(original).hexdigest()
        if digest != expected_digest:
            raise ValueError(f"reserved {role} input digest differs from the fixed export")
        episode, reader_receipt = read_continuous_probe_episode(path, role=role)
        if _bytes(path) != original or reader_receipt != len(original):
            raise ValueError(f"reserved {role} bytes changed across the existing reader")
        # Bind the returned episode itself, including non-history coordinates,
        # to the hashed payload; a byte-count receipt alone cannot establish it.
        if json.dumps(asdict(episode), sort_keys=True, allow_nan=False) != json.dumps(
            json.loads(original), sort_keys=True, allow_nan=False
        ):
            raise ValueError(f"reserved {role} decoded episode differs from the original bytes")
        binding = pick_cube_probe_binding(role)
        if (
            episode.role != binding.role
            or episode.context != binding.context
            or episode.episode_id != binding.episode_id
            or len(episode.transitions) != 18
            or episode.spec != cube_taskspace_spec()
        ):
            raise ValueError(f"reserved {role} identity,18rows or taskspace spec differs")
        episode = validate_action_history_probe_episode(episode, recipe)
        episodes.append(episode)
        receipts.append(
            {
                "path": str(path),
                "sha256": digest,
                "bytes": len(original),
                "role": episode.role,
                "context": episode.context,
                "episode_id": episode.episode_id,
                "rows": len(episode.transitions),
                "reader_receipt": reader_receipt,
            }
        )
    plan = ContinuousProbePlan(
        episodes[0],
        episodes[1],
        updates=1,
        diagnostics=True,
        zero_velocity_control=False,
    )
    return plan, receipts
