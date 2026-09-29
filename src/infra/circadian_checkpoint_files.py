"""Store trusted runner checkpoints as replaceable local files.

Inputs are app checkpoint payloads and a local path; loading returns the
payload after a byte-integrity check. Pickle is used for NumPy/Torch state,
so callers must never load checkpoints from untrusted sources.
"""

from __future__ import annotations

import copyreg
from hashlib import sha256
from io import BytesIO
import os
from pathlib import Path
import pickle
import tempfile
from types import MappingProxyType
from typing import Any

from src.app.fixed_feature_checkpoint import FixedFeatureCircadianCheckpoint
from src.app.continual_arrived_checkpoint import ArrivedRunnerCheckpoint
from src.app.continual_arrived_selection_checkpoint import ArrivedSelectionCheckpoint
from src.app.continual_checkpoint import ContinualRunnerCheckpoint
from src.app.continual_replay_policy_checkpoint import ReplayPolicyRunnerCheckpoint
from src.app.toy_checkpoint import ToyRunnerCheckpoint
from src.app.vision_checkpoint import VisionRunnerCheckpoint
from src.app.v14_trial_checkpoint import V14TrialPrefixCheckpoint

_MAGIC = b"CIRCADIAN_FIXED_FEATURE_CHECKPOINT_V1\n"
_TOY_MAGIC = b"CIRCADIAN_TOY_CHECKPOINT_V1\n"
_CONTINUAL_MAGIC = b"CIRCADIAN_CONTINUAL_CHECKPOINT_V1\n"
_ARRIVED_CONTINUAL_MAGIC = b"CIRCADIAN_ARRIVED_CONTINUAL_CHECKPOINT_V6\n"
_ARRIVED_SELECTION_MAGIC = b"CIRCADIAN_ARRIVED_SELECTION_CHECKPOINT_V7\n"
_REPLAY_POLICY_MAGIC = b"CIRCADIAN_REPLAY_POLICY_CHECKPOINT_V9\n"
_VISION_MAGIC = b"CIRCADIAN_VISION_CHECKPOINT_V1\n"
_V14_TRIAL_MAGIC = b"CIRCADIAN_V14_TRIAL_PREFIX_CHECKPOINT_V10\n"


class TrustedLocalCircadianCheckpointStore:
    """Replace a complete checkpoint file and reject accidental corruption."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> FixedFeatureCircadianCheckpoint:
        checkpoint = _load_payload(self.path, _MAGIC)
        if not isinstance(checkpoint, FixedFeatureCircadianCheckpoint):
            raise ValueError("checkpoint file payload type is incompatible")
        return checkpoint

    def save(self, checkpoint: FixedFeatureCircadianCheckpoint) -> None:
        if not isinstance(checkpoint, FixedFeatureCircadianCheckpoint):
            raise TypeError("checkpoint payload must be FixedFeatureCircadianCheckpoint")
        _save_payload(self.path, _MAGIC, checkpoint)


class TrustedLocalToyCheckpointStore:
    """Persist the whole toy runner state after each completed transaction."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> ToyRunnerCheckpoint:
        checkpoint = _load_payload(self.path, _TOY_MAGIC)
        if not isinstance(checkpoint, ToyRunnerCheckpoint):
            raise ValueError("checkpoint file payload type is incompatible")
        return checkpoint

    def save(self, checkpoint: ToyRunnerCheckpoint) -> None:
        if not isinstance(checkpoint, ToyRunnerCheckpoint):
            raise TypeError("checkpoint payload must be ToyRunnerCheckpoint")
        _save_payload(self.path, _TOY_MAGIC, checkpoint)


class TrustedLocalContinualCheckpointStore:
    """Persist one complete seed/phase transaction and prior seed reports."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> ContinualRunnerCheckpoint:
        checkpoint = _load_payload(self.path, _CONTINUAL_MAGIC)
        if not isinstance(checkpoint, ContinualRunnerCheckpoint):
            raise ValueError("checkpoint file payload type is incompatible")
        return checkpoint

    def save(self, checkpoint: ContinualRunnerCheckpoint) -> None:
        if not isinstance(checkpoint, ContinualRunnerCheckpoint):
            raise TypeError("checkpoint payload must be ContinualRunnerCheckpoint")
        _save_payload(self.path, _CONTINUAL_MAGIC, checkpoint)


class TrustedLocalArrivedCheckpointStore:
    """Persist only the separately typed v6 continual checkpoint."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> ArrivedRunnerCheckpoint:
        checkpoint = _load_payload(self.path, _ARRIVED_CONTINUAL_MAGIC)
        if not isinstance(checkpoint, ArrivedRunnerCheckpoint):
            raise ValueError("v6 checkpoint file payload type is incompatible")
        return checkpoint

    def save(self, checkpoint: ArrivedRunnerCheckpoint) -> None:
        if not isinstance(checkpoint, ArrivedRunnerCheckpoint):
            raise TypeError("checkpoint payload must be ArrivedRunnerCheckpoint")
        _save_payload(self.path, _ARRIVED_CONTINUAL_MAGIC, checkpoint)


class TrustedLocalArrivedSelectionCheckpointStore:
    """Persist one v7 candidate-manifest transaction as a trusted local file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> ArrivedSelectionCheckpoint:
        checkpoint = _load_payload(self.path, _ARRIVED_SELECTION_MAGIC)
        if not isinstance(checkpoint, ArrivedSelectionCheckpoint):
            raise ValueError("v7 selection checkpoint payload type is incompatible")
        return checkpoint

    def save(self, checkpoint: ArrivedSelectionCheckpoint) -> None:
        if not isinstance(checkpoint, ArrivedSelectionCheckpoint):
            raise TypeError("checkpoint payload must be ArrivedSelectionCheckpoint")
        _save_payload(self.path, _ARRIVED_SELECTION_MAGIC, checkpoint)


class TrustedLocalReplayPolicyCheckpointStore:
    """Persist only the separate v8 policy comparison's format-9 cursor."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> ReplayPolicyRunnerCheckpoint:
        checkpoint = _load_payload(self.path, _REPLAY_POLICY_MAGIC)
        if type(checkpoint) is not ReplayPolicyRunnerCheckpoint:
            raise ValueError("v8 policy checkpoint payload type is incompatible")
        return checkpoint

    def save(self, checkpoint: ReplayPolicyRunnerCheckpoint) -> None:
        if type(checkpoint) is not ReplayPolicyRunnerCheckpoint:
            raise TypeError("checkpoint payload must be ReplayPolicyRunnerCheckpoint")
        _save_payload(self.path, _REPLAY_POLICY_MAGIC, checkpoint)


class TrustedLocalVisionCheckpointStore:
    """Persist one unmatched vision run as a trusted replaceable local file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> VisionRunnerCheckpoint:
        checkpoint = _load_payload(self.path, _VISION_MAGIC)
        if not isinstance(checkpoint, VisionRunnerCheckpoint):
            raise ValueError("checkpoint file payload type is incompatible")
        return checkpoint

    def save(self, checkpoint: VisionRunnerCheckpoint) -> None:
        if not isinstance(checkpoint, VisionRunnerCheckpoint):
            raise TypeError("checkpoint payload must be VisionRunnerCheckpoint")
        _save_payload(self.path, _VISION_MAGIC, checkpoint)


class TrustedLocalV14TrialCheckpointStore:
    """Write one immutable completed-trial prefix with a format-10 header."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> V14TrialPrefixCheckpoint:
        checkpoint = _load_payload(self.path, _V14_TRIAL_MAGIC)
        if type(checkpoint) is not V14TrialPrefixCheckpoint:
            raise ValueError("v14 trial checkpoint payload type is incompatible")
        return checkpoint

    def save(self, checkpoint: V14TrialPrefixCheckpoint) -> None:
        if type(checkpoint) is not V14TrialPrefixCheckpoint:
            raise TypeError("checkpoint payload must be V14TrialPrefixCheckpoint")
        if self.path.exists():
            raise FileExistsError(f"v14 trial checkpoint already exists: {self.path}")
        # Why this: the sealed role container holds read-only mapping proxies.
        # Scope the reducer to this checkpoint rather than changing global pickle.
        buffer = BytesIO()
        writer = pickle.Pickler(buffer, protocol=pickle.HIGHEST_PROTOCOL)
        dispatch = copyreg.dispatch_table.copy()
        dispatch[type(MappingProxyType({}))] = _reduce_mapping_proxy
        writer.dispatch_table = dispatch
        writer.dump(checkpoint)
        _save_serialized_payload(self.path, _V14_TRIAL_MAGIC, buffer.getvalue())


def _reduce_mapping_proxy(value: Any) -> tuple[Any, tuple[dict[Any, Any]]]:
    return _restore_mapping_proxy, (dict(value),)


def _restore_mapping_proxy(value: dict[Any, Any]) -> Any:
    return MappingProxyType(value)


def _load_payload(path: Path, magic: bytes) -> object:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"checkpoint file cannot be read: {path}") from exc
    if not raw.startswith(magic):
        raise ValueError("checkpoint file header is corrupt or incompatible")
    checksum, separator, payload = raw[len(magic) :].partition(b"\n")
    if (
        separator != b"\n"
        or len(checksum) != 64
        or checksum != sha256(payload).hexdigest().encode("ascii")
    ):
        raise ValueError("checkpoint file checksum is corrupt")
    try:
        return pickle.loads(payload)
    except Exception as exc:
        raise ValueError("checkpoint file payload is corrupt or incompatible") from exc


def _save_payload(path: Path, magic: bytes, checkpoint: object) -> None:
    payload = pickle.dumps(checkpoint, protocol=pickle.HIGHEST_PROTOCOL)
    _save_serialized_payload(path, magic, payload)


def _save_serialized_payload(path: Path, magic: bytes, payload: bytes) -> None:
    content = magic + sha256(payload).hexdigest().encode("ascii") + b"\n" + payload
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        Path(temporary_path).unlink(missing_ok=True)
