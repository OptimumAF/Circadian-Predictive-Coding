"""Atomically store a local v14 run cursor and immutable trial checkpoints.

Inputs are already validated trial checkpoints and run identity. Outputs
are a hidden run-state directory and exact checkpoint bytes. This module
does not train, score, or decide whether a stored prefix is scientifically
valid; callers must run the app preflight after loading.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from importlib import import_module
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterator
from uuid import uuid4

from src.app.v14_trial_checkpoint import V14TrialPrefixCheckpoint
from src.core.run_manifest import validate_run_id
from src.infra.circadian_checkpoint_files import TrustedLocalV14TrialCheckpointStore


V14_RESUME_STATE_SCHEMA = "v14_trial_resume_state_v1"
_STATUSES = {"incomplete", "failed", "canceled", "completed"}


@dataclass(frozen=True)
class V14ResumeState:
    """One replaceable cursor; checkpoint files themselves are immutable."""

    schema_id: str
    run_id: str
    status: str
    environment: dict[str, object]
    config_sha256: str
    protocol_sha256: str
    capture_wake_diagnostics: bool
    next_trial_index: int
    next_cell: tuple[int, str] | None
    checkpoint_file: str | None
    checkpoint_sha256: str | None
    error_type: str | None


def _serialize_state(state: V14ResumeState) -> bytes:
    return (json.dumps(asdict(state), sort_keys=True, allow_nan=False, indent=2) + "\n").encode(
        "utf-8"
    )


def _write_state(path: Path, state: V14ResumeState) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=".run-state.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(_serialize_state(state))
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _read_state(path: Path) -> V14ResumeState:
    try:
        record: Any = json.loads(
            path.read_bytes(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value))
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("v14 resume state is missing or invalid JSON") from exc
    if not isinstance(record, dict) or set(record) != set(V14ResumeState.__dataclass_fields__):
        raise ValueError("v14 resume state fields differ")
    if isinstance(record["next_cell"], list):
        record["next_cell"] = tuple(record["next_cell"])
    state = V14ResumeState(**record)
    if (
        state.schema_id != V14_RESUME_STATE_SCHEMA
        or state.status not in _STATUSES
        or type(state.environment) is not dict
        or type(state.capture_wake_diagnostics) is not bool
        or type(state.next_trial_index) is not int
        or (
            state.next_cell is not None
            and (type(state.next_cell) is not tuple or len(state.next_cell) != 2)
        )
        or (state.checkpoint_file is None) != (state.checkpoint_sha256 is None)
        or (state.error_type is not None and type(state.error_type) is not str)
    ):
        raise ValueError("v14 resume state shape differs")
    return state


class V14ResumeFiles:
    """Own one hidden cursor directory for a public run ID."""

    def __init__(self, output_root: str | Path, run_id: str) -> None:
        self.output_root = Path(output_root)
        self.run_id = validate_run_id(run_id)
        self.directory = self.output_root / f".{self.run_id}.resume"
        self.state_path = self.directory / "run-state.json"
        self.lock_path = self.output_root / f".{self.run_id}.resume.lock"

    @contextmanager
    def claim(self) -> Iterator[None]:
        """Use an OS lock that releases even after process termination."""
        self.output_root.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+b") as stream:
            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            if os.name == "nt":
                import msvcrt

                try:
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError as exc:
                    raise RuntimeError("v14 resume run is already active") from exc
                try:
                    yield
                finally:
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl = import_module("fcntl")

                try:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as exc:
                    raise RuntimeError("v14 resume run is already active") from exc
                try:
                    yield
                finally:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def create(
        self,
        environment: dict[str, object],
        config_sha256: str,
        protocol_sha256: str,
        capture_wake_diagnostics: bool,
        first_cell: tuple[int, str],
    ) -> V14ResumeState:
        if self.directory.exists():
            raise FileExistsError(f"v14 resume state already exists: {self.directory}")
        state = V14ResumeState(
            V14_RESUME_STATE_SCHEMA,
            self.run_id,
            "incomplete",
            environment,
            config_sha256,
            protocol_sha256,
            capture_wake_diagnostics,
            0,
            first_cell,
            None,
            None,
            None,
        )
        stage = Path(
            tempfile.mkdtemp(prefix=f".{self.run_id}.resume.pending.", dir=self.output_root)
        )
        _write_state(stage / "run-state.json", state)
        if self.directory.exists():
            raise FileExistsError(f"v14 resume state already exists: {self.directory}")
        os.rename(stage, self.directory)
        return state

    def load(self) -> V14ResumeState:
        state = _read_state(self.state_path)
        if state.run_id != self.run_id:
            raise ValueError("v14 resume state run ID differs from directory")
        return state

    def write(self, state: V14ResumeState) -> None:
        if state.run_id != self.run_id or not self.directory.is_dir():
            raise ValueError("v14 resume state destination differs")
        _write_state(self.state_path, state)

    def mark(
        self, state: V14ResumeState, status: str, error_type: str | None = None
    ) -> V14ResumeState:
        if status not in _STATUSES:
            raise ValueError("v14 resume status differs")
        changed = replace(state, status=status, error_type=error_type)
        self.write(changed)
        return changed

    def commit(self, state: V14ResumeState, checkpoint: V14TrialPrefixCheckpoint) -> V14ResumeState:
        """Save a new immutable file first, then point the atomic cursor at it."""
        index = checkpoint.next_trial_index
        name = f"checkpoint-{index:02d}-{uuid4().hex}.bin"
        path = self.directory / name
        TrustedLocalV14TrialCheckpointStore(path).save(checkpoint)
        digest = sha256(path.read_bytes()).hexdigest()
        changed = replace(
            state,
            status="incomplete",
            next_trial_index=index,
            next_cell=checkpoint.next_cell,
            checkpoint_file=name,
            checkpoint_sha256=digest,
            error_type=None,
        )
        self.write(changed)
        return changed

    def load_checkpoint(self, state: V14ResumeState) -> V14TrialPrefixCheckpoint:
        name = state.checkpoint_file
        if (
            name is None
            or not re.fullmatch(
                rf"checkpoint-{state.next_trial_index:02d}-[0-9a-f]{{32}}\.bin", name
            )
            or type(state.checkpoint_sha256) is not str
        ):
            raise ValueError("v14 resume checkpoint identity differs")
        path = self.directory / name
        if path.is_symlink():
            raise ValueError("v14 resume checkpoint cannot be a symlink")
        try:
            actual = sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            raise ValueError("v14 resume checkpoint file is missing") from exc
        if actual != state.checkpoint_sha256:
            raise ValueError("v14 resume checkpoint SHA-256 differs")
        return TrustedLocalV14TrialCheckpointStore(path).load()
