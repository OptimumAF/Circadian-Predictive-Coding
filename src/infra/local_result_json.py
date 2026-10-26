"""Write completed dataclass reports to new finite local JSON files.

Inputs are a finished report and path. This adapter owns encoding only; it
does not run training, choose metrics, or open evaluation data.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np


def write_local_result_json(result: Any, path: str | Path) -> None:
    """Write a complete result once, refusing to replace an existing path."""
    write_local_json_payload(asdict(result), path)


def write_local_json_payload(record: Any, path: str | Path) -> None:
    """Write a finished JSON-safe record to a new local file."""
    payload = json.dumps(record, default=_encode_numpy, indent=2, allow_nan=False)
    with Path(path).open("x", encoding="utf-8") as output:
        output.write(payload + "\n")


def encode_bounded_local_json(record: Any, *, max_bytes: int) -> bytes:
    """Encode every value compactly; refuse oversize before opening a file.

    Why this: deeply nested scientific evidence can exceed an output allowance
    through indentation alone. Whitespace is disposable; evidence is not.
    The original pretty local writer remains unchanged for existing callers.
    """
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("local JSON byte cap must be an exact positive integer")
    payload = (
        json.dumps(record, default=_encode_numpy, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")
    if len(payload) > max_bytes:
        raise ValueError(f"local JSON is {len(payload)} bytes; exceeds {max_bytes}-byte output cap")
    return payload


def _encode_numpy(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"unsupported result JSON value: {type(value).__name__}")
