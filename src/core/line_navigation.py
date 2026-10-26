"""A deterministic two-action line environment for complete demonstration episodes.

Inputs are bounded integer starts/goals and legal actions; outputs are actual
next observations, rewards and termination. No learner, replay, RNG or IO.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class LineStep:
    observation: tuple[float, float]
    reward: float
    terminal: bool


class LineNavigation:
    def __init__(self, start: int, goal: int) -> None:
        if any(type(v) is not int or not -2 <= v <= 2 for v in (start, goal)) or start == goal:
            raise ValueError("line start/goal must be distinct native integers in -2..2")
        self._position, self._goal = start, goal

    @property
    def observation(self) -> tuple[float, float]:
        return self._position / 2, self._goal / 2

    @property
    def legal_actions(self) -> tuple[int, ...]:
        """Expose the current action set without applying or repairing an action."""
        if self._position == self._goal:
            return ()
        if self._position == -2:
            return (1,)
        if self._position == 2:
            return (0,)
        return (0, 1)

    def step(self, action: int) -> LineStep:
        if type(action) is not int or action not in (0, 1):
            raise ValueError("line action must be native integer 0 (left) or 1 (right)")
        if self._position == self._goal:
            raise ValueError("line episode is terminal")
        position = self._position + (1 if action else -1)
        if not -2 <= position <= 2:
            raise ValueError("line action crosses the boundary")
        self._position = position
        terminal = position == self._goal
        return LineStep(self.observation, float(terminal), terminal)
