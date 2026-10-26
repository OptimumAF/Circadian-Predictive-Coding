"""Bounded detached phase observations for the existing continuous stream.

Each callback receives a fork, not the stream learner. Returned finite JSON is
retained before integrity checks; callbacks never select or update stream work.
No model initialization, scheduling, training, persistence or simulator belongs
here. Callback computation is opaque and is not included in numerical work.
"""

from copy import deepcopy
import json
import math
from typing import Any, Callable

import numpy as np

from src.adapters.continuous_learnability_modes import (
    ContinuousLearner,
    ContinuousState,
    same_continuous_state,
)
from src.adapters.numpy_continuous_bc import (
    NumpyContinuousBCLearner,
    _snapshot as own_bc_snapshot,
)
from src.adapters.numpy_continuous_pc import (
    ContinuousPCSnapshot,
    NumpyContinuousPCLearner,
    _snapshot as own_pc_snapshot,
)

PhaseReadoutObserver = Callable[[ContinuousLearner, dict[str, Any]], dict[str, Any]]
_DEFINITION = "continuous_detached_phase_readouts_v1"
_PHASES = ("initial", "warm", "new", "warm_return")


def _metadata(value: dict[str, Any]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {
        "checkpoint_index",
        "phase",
        "updates",
        "training_examples",
    }:
        raise ValueError("phase metadata requires the four fixed checkpoint fields")
    for key in ("checkpoint_index", "updates", "training_examples"):
        if type(value[key]) is not int or not 0 <= value[key] <= 2**53 - 1:
            raise ValueError("phase metadata counters must be native nonnegative integers")
    index = value["checkpoint_index"]
    if index > 3 or type(value["phase"]) is not str or value["phase"] != _PHASES[index]:
        raise ValueError("phase metadata differs from its fixed checkpoint ordinal")
    return dict(value)


def _arrays(state: ContinuousState) -> tuple:
    values: tuple[np.ndarray, ...] = state.parameters + state.velocity
    if type(state) is ContinuousPCSnapshot and state.wake_state is not None:
        values += (state.wake_state.chemical_state, state.wake_state.plasticity_state)
    return values


def _numeric_bytes(state: ContinuousState) -> int:
    return sum(int(value.nbytes) for value in _arrays(state))


def _no_alias(first: ContinuousLearner, second: ContinuousLearner) -> None:
    if any(np.shares_memory(a, b) for a in _arrays(first._state) for b in _arrays(second._state)):
        raise RuntimeError("phase fork aliases its input parameters, momentum or wake")


def _finite_json(value: dict[str, Any]) -> tuple[dict[str, Any], int]:
    # Why: reject coercion, cycles and excessive nesting before serializing an
    # observer's untrusted result. The 64 KiB bound includes keys and strings.
    nodes = 0
    text_values = 0

    def check(item: Any, depth: int) -> None:
        nonlocal nodes, text_values
        nodes += 1
        if depth > 32 or nodes > 65536:
            raise ValueError("phase result exceeds bounded JSON structure")
        if type(item) is dict:
            for key, child in item.items():
                if type(key) is not str:
                    raise ValueError("phase result JSON keys must be native strings")
                text_values += len(key)
                if text_values > 65536:
                    raise ValueError("phase result exceeds bounded JSON text")
                check(child, depth + 1)
        elif type(item) is list:
            for child in item:
                check(child, depth + 1)
        elif type(item) is str:
            text_values += len(item)
            if text_values > 65536:
                raise ValueError("phase result exceeds bounded JSON text")
        elif type(item) is float:
            if not math.isfinite(item):
                raise ValueError("phase result must contain only finite JSON numbers")
        elif item is not None and type(item) not in (str, int, bool):
            raise ValueError("phase result must contain native JSON values")

    if type(value) is not dict:
        raise ValueError("phase observer must return an exact JSON dictionary")
    check(value, 0)
    encoded = json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"))
    size = len(encoded.encode("utf8"))
    if size > 65536:
        raise ValueError("phase observer result exceeds64KiB")
    return json.loads(encoded), size


def _error(value: BaseException) -> str:
    return (type(value).__name__ + ": " + str(value))[:512]


class PhaseReadouts:
    """Own at most four detached scalar observations and their paid prefix."""

    def __init__(self, observer: PhaseReadoutObserver) -> None:
        if not callable(observer):
            raise ValueError("phase observer must be callable")
        self._observer = observer
        self._records: list[dict[str, Any]] = []
        self._attempts = self._returns = self._validated = self._forks = 0
        self._failed = False
        self._work = dict(
            snapshot_copies=0,
            snapshot_numeric_bytes=0,
            fork_numeric_bytes=0,
            callback_result_bytes=0,
            unknown_callback_work=False,
        )

    @property
    def report(self) -> dict[str, Any]:
        return deepcopy(
            dict(
                definition=_DEFINITION,
                attempts=self._attempts,
                returns=self._returns,
                validated=self._validated,
                forks=self._forks,
                records=self._records,
                failed=self._failed,
                work=self._work,
            )
        )

    def _snapshot(self, learner: ContinuousLearner) -> ContinuousState:
        kind = type(learner)
        result: ContinuousState
        if kind is NumpyContinuousBCLearner and isinstance(learner, NumpyContinuousBCLearner):
            result = own_bc_snapshot(learner._state, learner._policy)
        elif kind is NumpyContinuousPCLearner and isinstance(learner, NumpyContinuousPCLearner):
            result = own_pc_snapshot(learner._state, learner._policy)
        else:
            raise ValueError("phase readout requires an exact native BP or PC learner")
        if type(learner) is not kind:
            raise RuntimeError("phase readout owner type changed")
        self._work["snapshot_copies"] += 1
        self._work["snapshot_numeric_bytes"] += _numeric_bytes(result)
        return result

    def _unchanged(self, learner: ContinuousLearner, expected: ContinuousState) -> None:
        observed = self._snapshot(learner)
        if type(expected) is not type(observed):
            raise RuntimeError("phase observer changed its native learner family")
        # Equality revalidates and owns both complete snapshots. Count returned
        # logical array payload copies, not Python objects or peak RSS.
        unchanged = same_continuous_state(expected, observed)
        self._work["snapshot_copies"] += 2
        self._work["snapshot_numeric_bytes"] += _numeric_bytes(expected) + _numeric_bytes(observed)
        if not unchanged:
            raise RuntimeError("phase observer changed a frozen learner state")

    def observe(
        self, readout: ContinuousLearner, metadata: dict[str, Any], guard: Callable[[], None]
    ) -> None:
        if self._failed:
            raise RuntimeError("a failed phase observer cannot resume")
        entry: dict[str, Any] | None = None
        callback_error: BaseException | None = None
        detached: ContinuousLearner | None = None
        expected: ContinuousState | None = None
        try:
            owned = _metadata(metadata)
            if owned["checkpoint_index"] != len(self._records) or len(self._records) >= 4:
                raise ValueError("phase observations require the next fixed checkpoint")
            if not callable(guard):
                raise ValueError("phase readout guard must be callable")
            expected = self._snapshot(readout)
            if (owned["updates"], owned["training_examples"]) != (
                expected.completed_updates,
                expected.training_examples,
            ):
                raise ValueError("phase metadata counters differ from its native learner")
            guard()
            self._unchanged(readout, expected)
            detached = readout.fork()
            self._forks += 1
            if type(detached) is not type(readout) or detached is readout:
                raise RuntimeError("phase fork must be a distinct learner of the exact same type")
            self._unchanged(detached, expected)
            self._unchanged(readout, expected)
            _no_alias(readout, detached)
            self._work["fork_numeric_bytes"] += _numeric_bytes(expected)
            # Fork is opaque too: external stream owners must still be valid
            # immediately before any observer work is admitted.
            guard()
            if _metadata(metadata) != owned:
                raise RuntimeError("phase fork changed checkpoint metadata")
            self._unchanged(readout, expected)
            self._unchanged(detached, expected)
            _no_alias(readout, detached)
            passed = deepcopy(owned)
            entry = dict(metadata=deepcopy(owned))
            self._records.append(entry)
            self._attempts += 1
            self._work["unknown_callback_work"] = True
            try:
                returned = self._observer(detached, passed)
                self._returns += 1
                result, size = _finite_json(returned)
                entry["result"] = result
                self._work["callback_result_bytes"] += size
            except BaseException as failure:
                callback_error = failure
                entry["error"] = _error(failure)
                raise
            finally:
                # The root guard is paid even when the observer raises. Keep the
                # original callback exception and record a secondary guard error.
                try:
                    guard()
                    if _metadata(metadata) != owned or _metadata(passed) != owned:
                        raise RuntimeError("phase observer changed checkpoint metadata")
                    self._unchanged(readout, expected)
                    self._unchanged(detached, expected)
                    _no_alias(readout, detached)
                except BaseException as failure:
                    if callback_error is not None:
                        entry["secondary_guard_error"] = _error(failure)
                    else:
                        raise
            self._validated += 1
            entry["validated"] = True
        except BaseException as failure:
            self._failed = True
            if entry is not None and "error" not in entry:
                entry["error"] = _error(failure)
            raise
