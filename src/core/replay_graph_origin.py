"""Borrowed replay inventory ports for already identity-bound native state roots.

Rows are original references; byte counts measure replay arrays only, not full
model copies, heap or RSS. Callers retain provenance, admission, consent and
publication responsibilities. These ports never grant authority or copy data.
"""

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ReplayGraphPorts:
    rows: Callable[[object, int], tuple[object, ...]]
    payload_bytes: Callable[[object, int, int], int]

    def __post_init__(self) -> None:
        if not callable(self.rows) or not callable(self.payload_bytes):
            raise ValueError("replay graph inventory ports must be callable")
