"""Explicit interruption fixture after genuine native training, before return.

Inputs are an owned native model, actual session-authority reader, work counters
and trusted emit/wait callbacks. Output is full posttrain telemetry alongside the
still-pending authority. Why never return: the session must not save, predict or
complete C after this boundary. Telemetry confers no completion/recovery authority.
This wrapper neither trains extra steps nor writes checkpoints or resumes work.
"""

from hashlib import sha256
from math import isfinite
from typing import Any, Callable, NoReturn

from src.adapters.coupled_checkpoint_codec import CoupledCheckpointCodec
from src.adapters.native_transfer_protocol import BINDING, LIMITS
from src.adapters.native_transfer_session import _NativeModel
from src.core.coordinator_transfer import _authority
from src.core.coupled_energy import Array
from src.core.coupled_predictive_coding import CoupledSnapshot
from src.core.recovery_authority import AuthorityRecord
from src.infra.recovery_authority_codec import encode_authority


class PausedNativeModel:
    """Trusted DI delegate; a paused native train result can never escape."""

    def __init__(
        self,
        model: _NativeModel,
        authority: Callable[[], AuthorityRecord],
        counts: dict[str, int],
        *,
        emit: Callable[[dict[str, Any]], None],
        wait: Callable[[], object],
    ) -> None:
        if not callable(authority) or not callable(emit) or not callable(wait):
            raise ValueError("native pause requires trusted authority, emit and wait callbacks")
        if type(counts) is not dict or any(
            type(key) is not str or type(value) is not int or value < 0
            for key, value in counts.items()
        ):
            raise ValueError(
                "native pause counts require native string/nonnegative integer entries"
            )
        self._model, self._authority, self._counts = model, authority, counts
        self._emit, self._wait = emit, wait

    def snapshot_state(self) -> CoupledSnapshot:
        return self._model.snapshot_state()

    def predict_proba(self, features: Array) -> Array:
        return self._model.predict_proba(features)

    def train_epoch(self, *args: Any, **kwargs: Any) -> NoReturn:
        result = self._model.train_epoch(*args, **kwargs)
        if type(result.energy) not in (int, float):
            raise ValueError("paused native energy requires a finite native number")
        try:
            energy = float(result.energy)
        except OverflowError:
            raise ValueError("paused native energy is not representable") from None
        if not isfinite(energy):
            raise ValueError("paused native energy must be finite")
        payload = CoupledCheckpointCodec().encode(
            self._model.snapshot_state(), binding=BINDING, limits=LIMITS
        )
        authority = self._authority()
        _authority(authority)
        metadata = authority.metadata
        if (
            metadata.stopped
            or not metadata.uncertain_work
            or metadata.usage.updates_admitted != metadata.usage.updates_completed + 1
        ):
            raise ValueError("native pause requires the actual single pending update authority")
        self._emit(
            dict(
                event="native_paused",
                authority_utf8=encode_authority(authority).decode("utf-8"),
                payload_utf8=payload.decode("utf-8"),
                digest=sha256(payload).hexdigest(),
                energy=energy,
                budget=dict(self._counts),
            )
        )
        self._wait()
        raise RuntimeError("paused native update must never return a train result")
