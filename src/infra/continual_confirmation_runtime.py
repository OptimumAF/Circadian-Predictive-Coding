"""Observe bounded optimizer execution without changing pinned model state.

Inputs are fixed worker limits, a started RSS sampler and a monotonic clock.
Outputs count actual successful calls outside rollback snapshots. No dataset,
score, source selection or artifact publication belongs to this boundary.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from functools import wraps
from math import isfinite
from time import monotonic
from typing import Any, Callable, Iterator
from unittest.mock import patch

from src.app.continual_confirmation_manifest import ConfirmationManifest
from src.core.backprop_mlp import BackpropMLP
from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork
from src.core.predictive_coding import PredictiveCodingNetwork
from src.shared.process_memory import ProcessRssSampler


class ConfirmationStopped(RuntimeError):
    def __init__(self, reason: str, executed_updates: int, elapsed_seconds: float) -> None:
        self.reason = reason
        self.executed_updates = executed_updates
        self.elapsed_seconds = elapsed_seconds
        super().__init__(
            f"confirmation incomplete: {reason} after {executed_updates} updates ({elapsed_seconds:.3f}s)"
        )


class ExecutionObserver:
    """Check live limits and count each baseline/wake/replay optimizer call."""

    def __init__(
        self,
        manifest: ConfirmationManifest,
        sampler: ProcessRssSampler,
        *,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self.manifest = manifest
        self.sampler = sampler
        self.clock = clock
        self.executed_updates = 0
        self.attempted_updates = 0
        self.by_model_kind = {"backprop": 0, "pc": 0, "circadian": 0}
        self._active = False
        self.started = self._clock()
        self._last_clock = self.started
        if type(manifest.max_optimizer_updates) is not int or manifest.max_optimizer_updates <= 0:
            raise ValueError("confirmation optimizer cap must be a positive integer")
        if type(manifest.wall_limit_seconds) is not int or manifest.wall_limit_seconds <= 0:
            raise ValueError("confirmation wall cap must be a positive integer")
        if type(manifest.max_process_rss_bytes) is not int or manifest.max_process_rss_bytes <= 0:
            raise ValueError("confirmation RSS cap must be a positive integer")
        if (
            type(manifest.rss_interval_seconds) is not float
            or not isfinite(manifest.rss_interval_seconds)
            or manifest.rss_interval_seconds <= 0
            or sampler.interval_seconds != manifest.rss_interval_seconds
        ):
            raise ValueError("confirmation RSS sampling interval differs")
        self.checkpoint()

    def _clock(self) -> float:
        value = self.clock()
        if type(value) not in (int, float) or not isfinite(value):
            raise ValueError("confirmation clock must be finite")
        return float(value)

    def elapsed(self) -> float:
        now = self._clock()
        if now < self._last_clock:
            raise ValueError("confirmation clock moved backwards")
        self._last_clock = now
        return now - self.started

    def checkpoint(self) -> None:
        elapsed = self.elapsed()
        if elapsed >= self.manifest.wall_limit_seconds:
            raise ConfirmationStopped("wall_limit", self.executed_updates, elapsed)
        current = self.sampler.sample()
        if type(current) is not int or current <= 0:
            raise RuntimeError("confirmation RSS unavailable")
        memory = self.sampler.snapshot()
        if memory.peak_bytes > self.manifest.max_process_rss_bytes:
            raise ConfirmationStopped("rss_limit", self.executed_updates, elapsed)

    def updates(self) -> dict[str, Any]:
        return {
            "attempted_updates": self.attempted_updates,
            "executed_updates": self.executed_updates,
            "by_model_kind": dict(self.by_model_kind),
        }

    def _wrap(self, original: Callable[..., Any], kind: str) -> Callable[..., Any]:
        @wraps(original)
        def observed(model: Any, *args: Any, **kwargs: Any) -> Any:
            self.checkpoint()
            if self.executed_updates >= self.manifest.max_optimizer_updates:
                raise ConfirmationStopped("optimizer_limit", self.executed_updates, self.elapsed())
            self.attempted_updates += 1
            result = original(model, *args, **kwargs)
            self.executed_updates += 1
            self.by_model_kind[kind] += 1
            # Why this: record successful work before a later resource stop;
            # accepted/rejected sleep restoration never restores this observer.
            self.checkpoint()
            return result

        return observed

    @contextmanager
    def observe_updates(self) -> Iterator[None]:
        if self._active:
            raise RuntimeError("confirmation observer is already active")
        self._active = True
        methods = (
            (BackpropMLP, "train_epoch", "backprop"),
            (PredictiveCodingNetwork, "train_epoch", "pc"),
            (CircadianPredictiveCodingNetwork, "_run_training_step", "circadian"),
        )
        try:
            with ExitStack() as stack:
                for model, name, kind in methods:
                    # The internal circadian seam covers wake and replay once,
                    # including inherited explicit-parent controls.
                    stack.enter_context(
                        patch.object(model, name, self._wrap(getattr(model, name), kind))
                    )
                self.checkpoint()
                yield
        finally:
            self._active = False
