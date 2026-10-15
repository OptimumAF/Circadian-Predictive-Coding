"""One trusted native-only session under existing original recovery accounting.

The coordinator reserves costs before checkpoint IO, complete model restoration,
training or prediction. Only a genuinely acknowledged completion is exported as
proof. Why retain the model only after success: failed restoration or uncertain
mutation must not become a usable continuation. Model factories are trusted DI;
canonical full-snapshot equality checks their restoration, not their provenance.
No journal creation, actor publication, process launch or recovery retry occurs.
"""

from dataclasses import dataclass, replace
from hashlib import sha256
from math import isfinite
import os
from pathlib import Path
from typing import Any, Callable, Protocol

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np

from src.adapters.coupled_checkpoint_codec import CoupledCheckpointCodec
from src.adapters.coupled_checkpoint_worker import _model
from src.adapters.native_transfer_protocol import (
    BINDING,
    LIMITS,
    FEATURES,
    TARGETS,
    completion_to_wire,
)
from src.app.recovery_coordinator import RecoveryCoordinator
from src.core.checkpoint_codec import CodecBinding, CodecLimits, require_codec_digest
from src.core.coordinator_transfer import _authority, _schema
from src.core.coupled_energy import Array
from src.core.coupled_predictive_coding import CoupledSnapshot, CoupledTrainResult
from src.core.recovery_admission import RecoveryManifest
from src.core.recovery_authority import AuthorityRecord
from src.core.recovery_completion import AuthorityCompletion
from src.core.recovery_coordination import RecoveryCosts, RecoveryRegistration
from src.core.recovery_observation import RecoveryObservationPort
from src.infra.checkpoint_files import BoundedCheckpointFileStore
from src.infra.recovery_authority_codec import encode_authority
from src.infra.sqlite_recovery_journal import SqliteRecoveryJournal


class _NativeModel(Protocol):
    def snapshot_state(self) -> CoupledSnapshot: ...
    def train_epoch(
        self,
        features: Array,
        targets: Array,
        learning_rate: float,
        inference_steps: int,
        inference_learning_rate: float,
    ) -> CoupledTrainResult: ...
    def predict_proba(self, features: Array) -> Array: ...


class CompletionJournal(SqliteRecoveryJournal):
    """Observe only acknowledged actual completion; all SQLite behavior stays native."""

    last_completion: AuthorityCompletion | None = None

    def complete(self, completion: AuthorityCompletion) -> bool:
        acknowledged = super().complete(completion)
        if acknowledged is True:
            self.last_completion = completion
        return acknowledged


def _controls(binding: CodecBinding, limits: CodecLimits) -> tuple[CodecBinding, CodecLimits]:
    _schema(binding, CodecBinding)
    _schema(limits, CodecLimits)
    owned_binding, owned_limits = replace(binding), replace(limits)
    if any(
        getattr(owned_limits, name) > cap
        for name, cap in (
            ("max_encoded_bytes", 65536),
            ("max_array_bytes", 8192),
            ("max_layers", 4),
            ("max_dimension", 16),
        )
    ):
        raise ValueError("native transfer codec controls exceed the fixed local ceilings")
    return owned_binding, owned_limits


def _path(value: Path) -> Path:
    if (
        not isinstance(value, Path)
        or not value.is_absolute()
        or "\x00" in str(value)
        or not value.name
        or not value.name.strip(".")
    ):
        raise ValueError("native transfer requires an absolute non-dot checkpoint Path")
    return Path(os.path.abspath(value))


@dataclass(frozen=True)
class _PreparedUpdate:
    model: _NativeModel
    before: bytes
    payload: bytes
    energy: float
    predictions: list[list[float]]


class NativeTransferSession:
    def __init__(
        self,
        known_authority: AuthorityRecord,
        journal_path: str | Path,
        observer: RecoveryObservationPort,
        anchor: RecoveryRegistration,
        worker: RecoveryRegistration,
        *,
        binding: CodecBinding = BINDING,
        limits: CodecLimits = LIMITS,
        model_factory: Callable[[CoupledSnapshot], _NativeModel] | None = None,
    ) -> None:
        _authority(known_authority)
        self._binding, self._limits = _controls(binding, limits)
        metadata = known_authority.metadata
        if (
            metadata.stopped
            or metadata.uncertain_work
            or metadata.usage.updates_admitted != metadata.usage.updates_completed
            or metadata.manifest.native_sha256 != metadata.manifest.payload_sha256
            or metadata.manifest.source_sha256 != self._binding.source_sha256
            or metadata.manifest.policy_sha256 != self._binding.policy_sha256
        ):
            raise ValueError(
                "native transfer requires independently known clean native-only bindings"
            )
        if model_factory is not None and not callable(model_factory):
            raise ValueError("native transfer model_factory must be a trusted callable")
        self._factory = _model if model_factory is None else model_factory
        self._codec = CoupledCheckpointCodec()
        self._model: _NativeModel | None = None
        self._failed, self._closed = False, False
        self._journal = CompletionJournal(journal_path, known_authority)
        self._coordinator = RecoveryCoordinator(
            known_authority, self._journal, observer, anchor, worker
        )

    def authority(self) -> AuthorityRecord:
        """Read the original journal floor, including inside charged trusted callbacks."""
        return self._journal.read()

    def _ready(self) -> None:
        if self._failed or self._closed:
            raise ValueError("native transfer session is closed or permanently failed")

    def _source_digest(self, digest: str) -> None:
        require_codec_digest(digest)
        manifest = self.authority().metadata.manifest
        if digest != manifest.native_sha256 or digest != manifest.payload_sha256:
            raise ValueError("native transfer input digest differs from original current authority")

    def _encode(self, snapshot: CoupledSnapshot) -> bytes:
        payload = self._codec.encode(snapshot, binding=self._binding, limits=self._limits)
        if tuple(weight.shape for weight in snapshot.parameters.hidden_weights) != (
            (2, 3),
            (3, 2),
        ) or snapshot.precisions != (1.0, 1.0):
            raise ValueError("native transfer requires frozen D2/H(3,2), precisions(1,1)")
        return payload

    def _load(self, source: Path, digest: str) -> tuple[CoupledSnapshot, bytes]:
        snapshot = BoundedCheckpointFileStore(source, self._codec).load(
            binding=self._binding, expected_sha256=digest, limits=self._limits
        )
        payload = self._encode(snapshot)
        if sha256(payload).hexdigest() != digest:
            raise ValueError("native transfer full checkpoint canonical digest differs")
        return snapshot, payload

    def _restored(self, snapshot: CoupledSnapshot, payload: bytes) -> _NativeModel:
        model = self._factory(snapshot)
        if self._encode(model.snapshot_state()) != payload:
            raise ValueError("native transfer factory did not restore the complete checkpoint")
        return model

    @staticmethod
    def _predictions(model: _NativeModel) -> list[list[float]]:
        prediction = model.predict_proba(np.array(FEATURES, dtype=np.float64))
        if (
            type(prediction) is not np.ndarray
            or prediction.dtype != np.float64
            or prediction.shape != (2, 1)
            or not np.all(np.isfinite(prediction))
            or np.any((prediction < 0) | (prediction > 1))
        ):
            raise ValueError(
                "native transfer prediction must be a finite float64 (2,1) probability matrix"
            )
        return prediction.tolist()

    def update(
        self, source: Path, target: Path, expected_source_sha256: str, expected_target_sha256: str
    ) -> dict[str, Any]:
        self._ready()
        source, target = _path(source), _path(target)
        if source == target:
            raise ValueError("native transfer update cannot overwrite its input checkpoint")
        require_codec_digest(expected_target_sha256)
        self._source_digest(expected_source_sha256)
        manifest = self.authority().metadata.manifest

        def prepare() -> _PreparedUpdate:
            model = self._model
            if model is None:
                snapshot, before = self._load(source, expected_source_sha256)
                model = self._restored(snapshot, before)
            else:
                before = self._encode(model.snapshot_state())
                if sha256(before).hexdigest() != expected_source_sha256:
                    raise ValueError("retained native transfer model differs from expected source")
            result = model.train_epoch(
                np.array(FEATURES, dtype=np.float64),
                np.array(TARGETS, dtype=np.float64),
                0.02,
                2,
                0.05,
            )
            if type(result.energy) not in (int, float):
                raise ValueError("native transfer energy requires a finite native number")
            try:
                energy = float(result.energy)
            except OverflowError:
                raise ValueError("native transfer energy is not representable") from None
            if not isfinite(energy):
                raise ValueError("native transfer energy must be finite")
            snapshot = model.snapshot_state()
            payload = self._encode(snapshot)
            if sha256(payload).hexdigest() != expected_target_sha256:
                raise ValueError(
                    "native transfer output differs from independently expected digest"
                )
            digest = BoundedCheckpointFileStore(target, self._codec).save(
                snapshot, binding=self._binding, limits=self._limits
            )
            if digest != expected_target_sha256:
                raise ValueError("native transfer saved digest differs from expected checkpoint")
            return _PreparedUpdate(model, before, payload, energy, self._predictions(model))

        def validate(value: _PreparedUpdate) -> RecoveryManifest:
            _, restored = self._load(target, expected_target_sha256)
            if (
                restored != value.payload
                or self._encode(value.model.snapshot_state()) != value.payload
            ):
                raise ValueError(
                    "native transfer reloaded checkpoint differs from full native result"
                )
            return replace(
                manifest,
                native_sha256=expected_target_sha256,
                payload_sha256=expected_target_sha256,
            )

        try:
            self._journal.last_completion = None
            value = self._coordinator.execute_native_update(
                RecoveryCosts(
                    updates=1, copy_bytes=4 * self._limits.max_encoded_bytes, checkpoint_attempts=1
                ),
                prepare,
                validate,
            )
            completion = self._journal.last_completion
            if completion is None:
                raise ValueError("native transfer lacks an acknowledged actual completion")
            result = dict(
                event="updated",
                authority_utf8=encode_authority(self.authority()).decode("utf-8"),
                completion=completion_to_wire(completion),
                before_payload_utf8=value.before.decode("utf-8"),
                payload_utf8=value.payload.decode("utf-8"),
                digest=expected_target_sha256,
                energy=value.energy,
                predictions=value.predictions,
            )
            self._model = value.model
            return result
        except BaseException:
            self._failed = True
            raise

    def restore(self, source: Path, expected_sha256: str) -> dict[str, Any]:
        self._ready()
        if self._model is not None:
            raise ValueError("native transfer restore requires no retained model")
        source = _path(source)
        self._source_digest(expected_sha256)

        def prepare() -> tuple[_NativeModel, bytes, list[list[float]]]:
            snapshot, payload = self._load(source, expected_sha256)
            model = self._restored(snapshot, payload)
            predictions = self._predictions(model)
            if self._encode(model.snapshot_state()) != payload:
                raise ValueError("native transfer prediction changed the complete restored model")
            return model, payload, predictions

        try:
            model, payload, predictions = self._coordinator.execute(
                RecoveryCosts(copy_bytes=2 * self._limits.max_encoded_bytes, checkpoint_attempts=1),
                prepare,
            )
            result = dict(
                event="restored",
                authority_utf8=encode_authority(self.authority()).decode("utf-8"),
                payload_utf8=payload.decode("utf-8"),
                digest=expected_sha256,
                predictions=predictions,
            )
            self._model = model
            return result
        except BaseException:
            self._failed = True
            raise

    def refuse_update(self) -> dict[str, Any]:
        self._ready()
        callbacks = 0

        def forbidden(*args: Any) -> Any:
            nonlocal callbacks
            callbacks += 1
            raise AssertionError("exhausted native transfer reached a forbidden callback")

        try:
            self._coordinator.execute_native_update(RecoveryCosts(updates=1), forbidden, forbidden)
        except ValueError as error:
            self._failed = True
            if str(error) != "original authority allowance exceeded" or callbacks:
                raise
            authority = self.authority()
            metadata = authority.metadata
            if (
                not metadata.stopped
                or metadata.uncertain_work
                or metadata.usage.updates_admitted != metadata.usage.updates_completed
            ):
                raise AssertionError(
                    "exhausted native transfer did not preserve clean stopped authority"
                )
            return dict(
                event="exhausted",
                authority_utf8=encode_authority(authority).decode("utf-8"),
                reason=str(error),
                callbacks=callbacks,
            )
        except BaseException:
            self._failed = True
            raise
        self._failed = True
        raise AssertionError("native transfer exhaustion unexpectedly succeeded")

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._coordinator.close()
