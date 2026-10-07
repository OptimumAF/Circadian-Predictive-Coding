"""Fixed synthetic A/B phase source for the v11 difficulty comparison.

Development fields are available on arrival. Final fields are generated
only when read by the explicit role-release boundary; this module neither
trains models nor chooses experimental settings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DifficultyPhaseSource:
    """A deterministic balanced phase with independent final RNG stream."""

    seed: int
    phase: str

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed < 0 or self.phase not in {"a", "b"}:
            raise ValueError("difficulty source requires a nonnegative seed and phase a/b")

    def _rows(self, *, final: bool) -> tuple[np.ndarray, np.ndarray]:
        offset = 1 if self.phase == "a" else 2
        rng = np.random.default_rng(self.seed * 100 + offset + (50 if final else 0))
        labels = np.tile(np.array([0.0, 1.0], dtype=np.float64), 20).reshape(-1, 1)
        signed = 2.0 * labels[:, 0] - 1.0
        centers = np.zeros((40, 2), dtype=np.float64)
        centers[:, 0 if self.phase == "a" else 1] = 1.2 * signed
        inputs = centers + rng.normal(0.0, 0.35, size=(40, 2))
        return inputs, labels

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
