"""Optional actual MiniGrid Memory boundary with owned public observations.

Inputs: explicit seed, role and <=16 native actions. Outputs: detached public
observations, charged step evidence and genuine terminated demonstration records.
No rendering, hidden-grid reads, policy, learner, replay or evaluation selection.
"""

from copy import deepcopy
from collections.abc import Callable
from importlib import import_module, metadata
from typing import Any

import numpy as np

from src.core.demonstration_episode import DemonstrationEpisode, DemonstrationTransition

MISSION = "go to the matching object at the end of the hallway"
ENVIRONMENT_ID = "MiniGrid-MemoryS7-v0"
ENCODING_ID = "minigrid_memory_public_image_direction_148_v1"
VERSIONS = {"minigrid": "3.1.0", "gymnasium": "1.2.3", "pygame-ce": "2.5.7"}
ACTION_NAMES = ("left", "right", "forward", "pickup", "drop", "toggle", "done")


def encode_memory_observation(observation: dict[str, Any]) -> tuple[float, ...]:
    """Encode the original Memory contract, retaining its fixed mission check."""
    return _encode_observation(observation, MISSION)


def _encode_observation(observation: dict[str, Any], mission: str) -> tuple[float, ...]:
    """Preserve all 147 image channels and direction; omit only fixed mission text.

    Why this: channel normalization fits existing bounded episode coordinates
    without cropping the cue or reading an unobservable target. C order is the
    native array order, with direction last. All returned scalars are native floats.
    """
    if type(observation) is not dict or set(observation) != {"image", "direction", "mission"}:
        raise ValueError("observation requires exactly image, direction and mission")
    image = observation["image"]
    if type(image) is not np.ndarray or image.dtype != np.uint8 or image.shape != (7, 7, 3):
        raise ValueError("Memory image must be a uint8 array with shape (7, 7, 3)")
    direction = observation["direction"]
    if type(direction) is not int or not 0 <= direction <= 3:
        raise ValueError("direction must be a native integer in 0..3")
    if type(observation["mission"]) is not str or observation["mission"] != mission:
        raise ValueError("mission must equal the fixed MiniGrid instruction")
    scales = np.array([10.0, 5.0, 2.0], dtype=np.float64)
    if np.any(image > scales):
        raise ValueError("image object/color/state exceed the pinned encoding ranges")
    values = (2.0 * image.astype(np.float64) / scales - 1.0).ravel(order="C")
    return (*map(float, values), 2.0 * direction / 3.0 - 1.0)


def _raw_observation(observation: dict[str, Any]) -> dict[str, Any]:
    return dict(
        image=observation["image"].tolist(),
        direction=observation["direction"],
        mission=observation["mission"],
    )


def _require_minigrid() -> Any:
    # Imports remain lazy: base installations and deterministic boundary fixtures
    # do not need Gymnasium, pygame, rendering, or a window.
    try:
        for name, version in VERSIONS.items():
            if metadata.version(name) != version:
                raise ValueError(f"{name} must be pinned to {version}")
        gym = import_module("gymnasium")
        import_module("minigrid")  # Registers the actual environment IDs.
    except (ModuleNotFoundError, metadata.PackageNotFoundError) as error:
        raise RuntimeError(
            "install optional dependencies using the minigrid extra from your local Circadian "
            "wheel; in a checkout: python -m pip install -r requirements-minigrid.txt"
        ) from error
    return gym


def _make_memory_environment(max_steps: int) -> Any:
    return _require_minigrid().make(ENVIRONMENT_ID, max_steps=max_steps, render_mode=None)


def _text(value: str, name: str, limit: int) -> None:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or len(value.encode("utf8")) > limit
    ):
        raise ValueError(f"{name} must be trimmed nonempty text within {limit} UTF-8 bytes")


class _MiniGridEpisode:
    """One reset, one step per action, no reuse after ending or an uncertain fault."""

    def __init__(
        self,
        seed: int,
        max_steps: int = 16,
        *,
        role: str = "train",
        episode_id: str = "minigrid-memory",
        context: str = "memory",
        mission: str,
        environment_id: str,
        encoding_id: str,
        factory: Callable[[int], Any],
    ) -> None:
        if type(seed) is not int or not 0 <= seed < 2**31:
            raise ValueError("seed must be a native integer in 0..2**31-1")
        if type(max_steps) is not int or not 1 <= max_steps <= 16:
            raise ValueError("max_steps must be a native integer in 1..16")
        if type(role) is not str or role not in (
            "train",
            "final_test",
            "probe_train",
            "probe_test",
        ):
            raise ValueError("role must be train, final_test, probe_train or probe_test")
        _text(episode_id, "episode_id", 64)
        _text(context, "context", 32)
        self._closed = False
        self._ended = False
        self._failed = False
        self._mission = mission
        self._report: dict[str, Any] = dict(
            environment_id=environment_id,
            encoding_id=encoding_id,
            versions=dict(VERSIONS),
            seed=seed,
            max_steps=max_steps,
            role=role,
            episode_id=episode_id,
            context=context,
            observation_dim=148,
            num_classes=7,
            action_names=ACTION_NAMES,
            reset_calls=0,
            environment_steps_attempted=0,
            environment_steps=0,
            step_accounting="environment_steps counts normal native returns before payload validation",
            failure_work_may_be_unobserved=False,
            status="running",
            closed=False,
            terminated=False,
            truncated=False,
            total_return=0.0,
            transitions=[],
        )
        self._environment = factory(max_steps)
        try:
            if self._environment.action_space.n != 7:
                raise ValueError("Memory environment must expose seven native actions")
            self._report["reset_calls"] += 1
            observation, _ = self._environment.reset(seed=seed)
            self._observation = _encode_observation(observation, self._mission)
            self._report["initial_observation"] = _raw_observation(observation)
        except Exception:
            self.close()
            raise

    @property
    def observation(self) -> tuple[float, ...]:
        return self._observation

    @property
    def legal_actions(self) -> tuple[int, ...]:
        # All seven actions are valid inputs, including blocked moves/no-ops.
        # Do not infer a mask from the private grid or desired outcome.
        return () if self._ended or self._closed or self._failed else tuple(range(7))

    @property
    def terminal(self) -> bool:
        return self._ended

    @property
    def report(self) -> dict[str, Any]:
        return deepcopy(self._report)

    def step(self, action: int) -> dict[str, Any]:
        if self._ended or self._closed or self._failed:
            raise ValueError("episode ended, closed or failed; no further steps admitted")
        if type(action) is not int or action not in range(7):
            raise ValueError("action must be a native integer in 0..6")
        record: dict[str, Any] = dict(
            observation=self._observation,
            action=action,
            attempted=True,
            applied=None,
            next_observation=None,
            reward=None,
            terminated=None,
            truncated=None,
        )
        self._report["transitions"].append(record)
        self._report["environment_steps_attempted"] += 1
        try:
            result = self._environment.step(action)
            # Charge normal returns BEFORE validation. A malformed result is not
            # free work; an exception may have changed the environment, so no retry.
            self._report["environment_steps"] += 1
            record["applied"] = True
            observation, reward, terminated, truncated, _ = result
            if type(terminated) is not bool or type(truncated) is not bool:
                raise ValueError("environment end flags must be native bools")
            if type(reward) not in (int, float) or not np.isfinite(reward) or not -1 <= reward <= 1:
                raise ValueError("environment reward must be finite in [-1, 1]")
            record.update(reward=float(reward), terminated=terminated, truncated=truncated)
            self._report["total_return"] += float(reward)
            self._report.update(terminated=terminated, truncated=truncated)
            self._ended = terminated or truncated
            self._observation = _encode_observation(observation, self._mission)
            record.update(
                next_observation=self._observation,
                raw_next_observation=_raw_observation(observation),
            )
            if not self._ended and self._report["environment_steps"] == self._report["max_steps"]:
                raise ValueError("environment failed to signal its declared max_steps boundary")
            if self._ended:
                self._report["status"] = "complete"
            return dict(
                observation=self._observation,
                reward=float(reward),
                terminated=terminated,
                truncated=truncated,
            )
        except Exception as error:
            self._failed = True
            self._report["failure_work_may_be_unobserved"] = record["applied"] is None
            self._report.update(status="failed", reason=f"{type(error).__name__}: {error}")
            raise

    def to_demonstration(self) -> DemonstrationEpisode:
        """Expose caller-scripted actions, never claim they are expert/successful.

        Truncation cannot be converted: DemonstrationEpisode requires a real last
        terminal transition. Its existing role gate separately prevents final replay.
        """
        report = self._report
        if report["status"] != "complete" or not report["terminated"] or report["truncated"]:
            raise ValueError("demonstration requires actual termination without truncation")
        transitions = tuple(
            DemonstrationTransition(
                i,
                row["observation"],
                row["action"],
                row["reward"],
                row["next_observation"],
                row["terminated"],
                7,
            )
            for i, row in enumerate(report["transitions"])
        )
        return DemonstrationEpisode(
            report["episode_id"], report["context"], report["role"], transitions, 7
        )

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._report["closed"] = True
            self._environment.close()

    def __enter__(self) -> "_MiniGridEpisode":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class MiniGridMemory(_MiniGridEpisode):
    """Original public Memory constructor and report schema, unchanged.

    Why this shared lifecycle: the second actual MiniGrid task needs the same
    exact step/failure/role handling. Only factories and fixed missions differ.
    """

    def __init__(
        self,
        seed: int,
        max_steps: int = 16,
        *,
        role: str = "train",
        episode_id: str = "minigrid-memory",
        context: str = "memory",
    ) -> None:
        super().__init__(
            seed,
            max_steps,
            role=role,
            episode_id=episode_id,
            context=context,
            mission=MISSION,
            environment_id=ENVIRONMENT_ID,
            encoding_id=ENCODING_ID,
            factory=_make_memory_environment,
        )
