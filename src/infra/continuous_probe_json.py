"""Bounded JSON input for caller-reserved continuous probe episodes.

Input is the exact dataclass-asdict episode schema and its required reserved role.
Output is an owned core episode and bytes read. This reader never relabels data,
qualifies collection, grants runtime release, selects probes, fits or trains.
Why: probe inputs are separate from the existing qualified TRAIN teacher format.
"""

import json
from pathlib import Path
from typing import Any

from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)


MAX_PROBE_FILE_BYTES = 2 * 1024**2


def _unique_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"probe JSON has duplicate field {name}")
        result[name] = value
    return result


def _reject_constant(value):
    raise ValueError(f"probe JSON requires finite numbers, got {value}")


def _fields(value: Any, expected: tuple[str, ...], name: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError(f"probe {name} requires exactly {', '.join(expected)}")
    return value


def _vector(value, name, maximum):
    if type(value) is not list or not 1 <= len(value) <= maximum:
        raise ValueError(f"probe {name} requires a JSON list with1..{maximum} entries")
    return tuple(value)


def _transition(value):
    row = _fields(
        value,
        (
            "step",
            "observation",
            "action",
            "reward",
            "next_observation",
            "terminated",
            "truncated",
        ),
        "transition",
    )
    return ContinuousTransition(
        row["step"],
        _vector(row["observation"], "observation", 1024),
        _vector(row["action"], "action", 16),
        row["reward"],
        _vector(row["next_observation"], "next_observation", 1024),
        row["terminated"],
        row["truncated"],
    )


def read_continuous_probe_episode(path: Path, *, role: str) -> tuple[ContinuousEpisode, int]:
    """Decode only the requested reserved role; no implicit conversion or filling."""
    if type(role) is not str or role not in ("probe_train", "probe_test"):
        raise ValueError("probe reader requires exact probe_train or probe_test role")
    with path.open("rb") as handle:
        payload = handle.read(MAX_PROBE_FILE_BYTES + 1)
    if len(payload) > MAX_PROBE_FILE_BYTES:
        raise ValueError("probe episode exceeds2MiB per-file allowance")
    try:
        raw = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
        data = _fields(raw, ("episode_id", "context", "role", "spec", "transitions"), "episode")
        if type(data["role"]) is not str or data["role"] != role:
            raise ValueError(f"probe input requires declared {role}, never relabels other roles")
        spec = _fields(
            data["spec"],
            (
                "observation_dim",
                "action_units",
                "lower_bounds",
                "upper_bounds",
            ),
            "spec",
        )
        declared = ContinuousActionSpec(
            spec["observation_dim"],
            _vector(spec["action_units"], "action_units", 16),
            _vector(spec["lower_bounds"], "lower_bounds", 16),
            _vector(spec["upper_bounds"], "upper_bounds", 16),
        )
        records = _vector(data["transitions"], "transitions", 128)
        episode = ContinuousEpisode(
            data["episode_id"],
            data["context"],
            data["role"],
            declared,
            tuple(_transition(row) for row in records),
        )
    except (UnicodeError, RecursionError, OverflowError) as error:
        raise ValueError("probe input must be bounded finite UTF-8 episode JSON") from error
    return episode, len(payload)
