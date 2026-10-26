"""Bounded ambient-light IO for the existing single-environment episode port.

Owns original readback and paid native read/write/refresh counters. The caller
supplies a pure binding/identity check. No physics, reset, encoding, teacher,
learner or retry is implemented here; refreshed observations go to the old port.
"""

from collections.abc import Callable
from dataclasses import dataclass, fields
import math
from typing import Any

import numpy as np

from src.core.pick_cube_lighting_context import (
    PickCubeLightingContext,
    validate_pick_cube_lighting_context,
)


@dataclass(frozen=True)
class ManiSkillLightingReadback:
    context: PickCubeLightingContext
    source_ambient_rgb: tuple[float, ...]
    applied_ambient_rgb: tuple[float, ...]


@dataclass(frozen=True)
class ManiSkillLightingWork:
    read_attempts: int
    read_returns: int
    write_attempts: int
    write_returns: int
    refresh_attempts: int
    refresh_returns: int
    discarded_reset_observations: int


def validate_lighting_readback(
    value: ManiSkillLightingReadback, context: PickCubeLightingContext
) -> ManiSkillLightingReadback:
    """Own original typed measurements without touching a live scene."""
    expected = validate_pick_cube_lighting_context(context)
    if type(value) is not ManiSkillLightingReadback:
        raise ValueError("expected exact owned lighting readback")
    if validate_pick_cube_lighting_context(value.context) != expected:
        raise ValueError("lighting readback context changed")
    source, target = value.source_ambient_rgb, value.applied_ambient_rgb
    if any(
        type(group) is not tuple
        or len(group) != 3
        or any(
            type(item) is not float
            or not math.isfinite(item)
            or not 0.0 < item <= 1.0
            or float(np.float32(item)) != item
            for item in group
        )
        for group in (source, target)
    ) or any(t != s * expected.ambient_multiplier for s, t in zip(source, target, strict=True)):
        raise ValueError("lighting readback requires exact native float32 half RGB tuples")
    return ManiSkillLightingReadback(expected, tuple(source), tuple(target))


def validate_lighting_work(value: ManiSkillLightingWork) -> ManiSkillLightingWork:
    """Copy partial paid work; complete lifecycle checks belong to the caller."""
    if type(value) is not ManiSkillLightingWork or any(
        type(getattr(value, field.name)) is not int or not 0 <= getattr(value, field.name) < 2**63
        for field in fields(ManiSkillLightingWork)
    ):
        raise ValueError("lighting work requires bounded exact integer counters")
    if (
        not value.read_returns <= value.read_attempts
        or not value.write_returns <= value.write_attempts <= 1
        or not value.refresh_returns <= value.refresh_attempts <= 1
        or value.discarded_reset_observations != 1
        or (value.write_attempts and value.read_returns < 1)
        or (value.refresh_attempts and (value.write_returns != 1 or value.read_returns < 2))
    ):
        raise ValueError("lighting work differs from the bounded setter/refresh lifecycle")
    return ManiSkillLightingWork(*(getattr(value, field.name) for field in fields(value)))


class ManiSkillAmbientLight:
    """One admitted scene; constant storage, no implicit setter or reset retry."""

    def __init__(
        self,
        context: PickCubeLightingContext,
        base: Any,
        scene: Any,
        require_identity: Callable[[Any, Any], None],
    ):
        self._context = validate_pick_cube_lighting_context(context)
        self._base, self._scene = base, scene
        self._require_identity = require_identity
        self._reads = self._read_returns = self._writes = self._write_returns = 0
        self._refreshes = self._refresh_returns = 0
        self._source: tuple[float, ...] | None = None
        self._target: tuple[float, ...] | None = None
        self._original: ManiSkillLightingReadback | None = None

    def _require(self) -> None:
        source, target = self._source, self._target
        base, scene = self._base, self._scene
        validate_pick_cube_lighting_context(self._context)
        self._require_identity(base, scene)
        validate_pick_cube_lighting_context(self._context)
        if self._base is not base or self._scene is not scene:
            raise ValueError("admitted lighting IO identities changed during identity access")
        if self._source != source or self._target != target:
            raise ValueError("original lighting measurements changed during identity access")

    def _read(self) -> tuple[float, ...]:
        source, target = self._source, self._target
        self._require()
        self._reads += 1
        value = self._scene.ambient_light
        self._require()
        if self._source != source or self._target != target:
            raise ValueError("original lighting measurements changed during native read")
        if (
            type(value) is not np.ndarray
            or value.dtype != np.dtype("float32")
            or value.shape != (3,)
            or not np.isfinite(value).all()
            or not ((value > 0.0) & (value <= 1.0)).all()
        ):
            raise ValueError(
                "ambient light must be three finite positive native float32 values <=1"
            )
        owned = tuple(float(item) for item in value)
        self._read_returns += 1
        return owned

    def apply_and_refresh(self) -> Any:
        """Discard the reset packet; explicitly pay for fresh pixels after setting."""
        if self._reads or self._writes or self._refreshes:
            raise ValueError("lighting intervention cannot be repeated")
        self._source = self._read()
        self._target = tuple(
            float(np.float32(item * self._context.ambient_multiplier)) for item in self._source
        )
        if any(
            not 0.0 < target <= 1.0 or target != source * self._context.ambient_multiplier
            for source, target in zip(self._source, self._target, strict=True)
        ):
            raise ValueError(
                "declared exact half light is outside representable positive float32 bounds"
            )
        source, target = self._source, self._target
        self._writes += 1
        self._scene.set_ambient_light(list(self._target))
        self._write_returns += 1
        self._require()
        if self._source != source or self._target != target:
            raise ValueError("original lighting measurements changed during native setter")
        self.verify()
        self._original = ManiSkillLightingReadback(
            validate_pick_cube_lighting_context(self._context), source, target
        )
        self._refreshes += 1
        observation = self._base.get_obs()
        self._refresh_returns += 1
        self._require()
        if self._source != source or self._target != target:
            raise ValueError("original lighting measurements changed during observation refresh")
        # Why: the old encoder verifies light again before admitting this packet;
        # a refresh callback cannot silently supply stale/default-context pixels.
        return observation

    def verify(self) -> None:
        source, target, original = self._source, self._target, self._original
        if original is not None and (
            source != original.source_ambient_rgb or target != original.applied_ambient_rgb
        ):
            raise ValueError("original lighting measurements were replaced")
        actual = self._read()
        if (
            target is None
            or actual != target
            or self._source != source
            or self._target != target
            or self._original is not original
        ):
            raise ValueError("observed ambient light differs from its admitted target")

    @property
    def readback(self) -> ManiSkillLightingReadback | None:
        """Return owned original measurements without another native read."""
        if self._original is None:
            return None
        if (
            self._source != self._original.source_ambient_rgb
            or self._target != self._original.applied_ambient_rgb
        ):
            raise ValueError("original lighting measurements were replaced")
        return ManiSkillLightingReadback(
            validate_pick_cube_lighting_context(self._context),
            self._original.source_ambient_rgb,
            self._original.applied_ambient_rgb,
        )

    @property
    def work(self) -> ManiSkillLightingWork:
        return ManiSkillLightingWork(
            self._reads,
            self._read_returns,
            self._writes,
            self._write_returns,
            self._refreshes,
            self._refresh_returns,
            1,
        )

    def preserve_failure(self, error: BaseException) -> None:
        """A failed constructor still exposes paid attempts on its original error."""
        setattr(error, "lighting_work", self.work)
        setattr(error, "lighting_source_ambient_rgb", self._source)
        setattr(error, "lighting_requested_ambient_rgb", self._target)
        # Do not replace an original failure if its metadata was also corrupted.
        try:
            setattr(error, "lighting_readback", self.readback)
        except ValueError:
            setattr(error, "lighting_readback", None)
