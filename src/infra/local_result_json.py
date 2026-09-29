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
    payload = json.dumps(asdict(result), default=_encode_numpy, indent=2, allow_nan=False)
    with Path(path).open("x", encoding="utf-8") as output:
        output.write(payload + "\n")


def _encode_numpy(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"unsupported result JSON value: {type(value).__name__}")
