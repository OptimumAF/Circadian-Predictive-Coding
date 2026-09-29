"""Generate fixed noisy/shifted v13 phase fields with deferred final rows.

Inputs are a seed, phase, and distribution condition. Outputs are balanced
development fields on arrival and independent final fields on release.
This module does not schedule training, decide sleep, or score outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TriggerPhaseSource:
    """Pair stationary or axis-shifted phases under the same row noise."""

    seed: int
    phase: str
    condition: str

    def __post_init__(self) -> None:
        if (
            type(self.seed) is not int
            or self.seed < 0
            or self.phase not in {"a", "b"}
            or self.condition not in {"stationary_noise", "axis_shift"}
        ):
            raise ValueError("v13 source requires a nonnegative seed, phase a/b, and condition")

    def _rows(self, *, final: bool) -> tuple[np.ndarray, np.ndarray]:
        phase_offset = 1 if self.phase == "a" else 2
        rng = np.random.default_rng(100 * self.seed + phase_offset + (50 if final else 0))
        targets = np.tile(np.array([0.0, 1.0], dtype=np.float64), 20).reshape(-1, 1)
        centers = np.zeros((40, 2), dtype=np.float64)
        axis = 1 if self.condition == "axis_shift" and self.phase == "b" else 0
        centers[:, axis] = 1.2 * (2.0 * targets[:, 0] - 1.0)
        return centers + rng.normal(0.0, 0.55, size=(40, 2)), targets

    @property
    def train_input(self) -> np.ndarray:
        return self._rows(final=False)[0]

    @property
    def train_target(self) -> np.ndarray:
        return self._rows(final=False)[1]

    @property
    def test_input(self) -> np.ndarray:
        return self._rows(final=True)[0]

    @property
    def test_target(self) -> np.ndarray:
        return self._rows(final=True)[1]
