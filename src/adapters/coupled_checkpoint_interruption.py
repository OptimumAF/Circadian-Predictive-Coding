"""Windows checkpoint-only interruption with a surviving native actor.

Native A/B fixtures are prepared before the authority session. Child workers
never train. Existing coordinator counters and physical-process registrations
govern checkpoint staging, handoff and read-only restoration; no new restart
authority or native-completion protocol is inferred from valid model bytes.
"""

from contextlib import ExitStack
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import stat
import sys
import sysconfig
import time
from typing import Any

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np

from src.adapters.coupled_checkpoint_codec import CoupledCheckpointCodec
from src.app.recovery_coordinator import RecoveryCoordinator
from src.core.checkpoint_codec import CodecBinding, CodecLimits
from src.core.coupled_circadian_predictive_coding import (
    CoupledCircadianConfig,
    CoupledCircadianPredictiveCodingNetwork,
)
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork
from src.core.recovery_admission import (
    RecoveryLimits,
    RecoveryManifest,
    RecoveryMetadata,
    RecoveryUsage,
)
from src.core.recovery_authority import AuthorityRecord
from src.core.recovery_coordination import RecoveryCosts
from src.infra.bounded_json_child import BoundedJsonChild
from src.infra.checkpoint_files import BoundedCheckpointFileStore
from src.infra.sqlite_recovery_journal import SqliteRecoveryJournal
from src.infra.windows_process_handles import WindowsProcessHandle, WindowsRecoveryApi
from src.infra.windows_recovery_observer import WindowsRecoveryObserver

FAMILIES = ("pc", "neutral_cpc", "active_cpc")
LIMITS = CodecLimits(65536, 8192, 4, 16)
BINDING = CodecBinding(
    sha256(b"supervised coupled checkpoint fixture source v1").hexdigest(),
    sha256(b"supervised coupled checkpoint fixture policy v1").hexdigest(),
)
FEATURES = ((0.2, -0.3), (-0.4, 0.6))


def describe_example() -> dict[str, Any]:
    return dict(
        protocol_id="coupled_checkpoint_interruption_engineering_v1",
        families=list(FAMILIES),
        native_training_calls=9,
        native_energy_evaluations=36,
        owned_child_launches=6,
        child_training_calls=0,
        seed=23,
        geometry=dict(input_dim=2, hidden_dims=[3, 2], precisions=[1.0, 1.0]),
        training=dict(
            role="train", features=[list(row) for row in FEATURES], targets=[[1.0], [0.0]]
        ),
        external_training_settings=dict(
            learning_rate=0.02, inference_steps=2, inference_learning_rate=0.05
        ),
        checkpoint_attempt_limit=2,
        cpu_threads=1,
        limits=asdict(LIMITS),
        binding=asdict(BINDING),
        fixture_notes="Nine native fixture updates precede authority sessions. Children only stage bytes or restore/predict A. Fixture bindings are consistency labels, not authentication.",
        limitations="Windows; surviving parent only. No resumed uncertain training, B publication, coordinator-loss, full lifecycle, power-loss or durable-owner recovery.",
    )


def _plain(value: Any) -> Any:
    return json.loads(json.dumps(value, default=lambda array: array.tolist(), allow_nan=False))


def _model(family: str) -> CoupledPredictiveCodingNetwork:
    if family == "pc":
        return CoupledPredictiveCodingNetwork(2, (3, 2), seed=23, precisions=(1.0, 1.0))
    config = (
        CoupledCircadianConfig.matched_pc_control()
        if family == "neutral_cpc"
        else CoupledCircadianConfig()
    )
    return CoupledCircadianPredictiveCodingNetwork(
        2, (3, 2), seed=23, precisions=(1.0, 1.0), config=config
    )


def _fixtures(family: str):
    actor = _model(family)
    x, y = np.asarray(FEATURES, dtype=np.float64), np.array([[1.0], [0.0]])
    for _ in range(2):
        actor.train_epoch(x.copy(), y.copy(), 0.02, 2, 0.05)
    snapshot_a = actor.snapshot_state()
    candidate = _model(family)
    candidate.restore_state(snapshot_a)
    candidate.train_epoch(x.copy(), y.copy(), 0.02, 2, 0.05)
    return actor, snapshot_a, candidate.snapshot_state(), x


def _new_child(stack: ExitStack) -> BoundedJsonChild:
    executable = getattr(sys, "_base_executable", None)
    if type(executable) is not str or not Path(executable).is_file():
        raise ValueError("checkpoint example requires the existing physical base interpreter")
    repository = Path(__file__).resolve().parents[2]
    # A Windows venv redirector's PID is not the physical Python worker. Launch
    # the base interpreter directly with this environment's existing imports.
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(repository), sysconfig.get_path("purelib")))
    child = stack.enter_context(
        BoundedJsonChild(
            [executable, "-u", "-B", "-X", "utf8", "-m", "src.adapters.coupled_checkpoint_worker"],
            cwd=repository,
            env=env,
        )
    )
    hello = child.receive()
    if hello != dict(event="hello", pid=child.pid, version=list(sys.version_info[:3])):
        raise ValueError("worker hello differs from owned physical PID/interpreter")
    return child


def _request(path: Path, digest: str, *, payload: bytes | None = None) -> dict[str, Any]:
    request = dict(
        op="read" if payload is None else "stage",
        path=str(path),
        binding=asdict(BINDING),
        limits=asdict(LIMITS),
        expected_sha256=digest,
    )
    if payload is not None:
        request["payload_utf8"] = payload.decode("utf-8")
    return request


def _bytes(path: Path) -> bytes:
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("example checkpoint/staging path must be a regular file")
    with path.open("rb") as stream:
        raw = stream.read(LIMITS.max_encoded_bytes + 1)
    if len(raw) > LIMITS.max_encoded_bytes:
        raise ValueError("example checkpoint exceeds encoded bound")
    return raw


def _staging_path(path: Path, event: dict[str, Any], expected: bytes) -> Path:
    if set(event) != {"event", "temporary", "digest"} or event["event"] != "staged":
        raise ValueError("worker did not report the staged checkpoint event")
    name = event["temporary"]
    if (
        type(name) is not str
        or Path(name).name != name
        or not name.startswith(f".{path.name}.")
        or not name.endswith(".tmp")
        or event["digest"] != sha256(expected).hexdigest()
    ):
        raise ValueError("worker staging event differs from owned file/digest")
    temporary = path.parent / name
    if _bytes(temporary) != expected:
        raise ValueError("actual staged bytes differ from the independently prepared B")
    return temporary


def _authority(observation, anchor, worker, digest: str, family: str) -> AuthorityRecord:
    absent = sha256(b"absent component in checkpoint-only engineering fixture").hexdigest()
    manifest = RecoveryManifest(
        BINDING.source_sha256,
        BINDING.policy_sha256,
        digest,
        digest,
        absent,
        absent,
        absent,
        digest,
        absent,
    )
    metadata = RecoveryMetadata(
        1,
        f"checkpoint-writer-{family}-{anchor.pid}",
        observation.clock_epoch,
        "writer-one",
        0,
        0,
        observation.now_ns,
        observation.now_ns,
        0,
        manifest,
        RecoveryLimits(0, 60_000_000_000, 4 * LIMITS.max_encoded_bytes, 0, 2, 512 * 1024**2),
        RecoveryUsage(0, 0, 0, 0, 0, observation.peak_rss_bytes),
        False,
        False,
    )
    return AuthorityRecord(metadata, anchor, worker)


def _family_example(directory: Path, family: str) -> dict[str, Any]:
    # Keep the ordered ownership transaction under one ExitStack; launch,
    # validation and fixture helpers are separate so resource lifetimes stay visible.
    # All native update work is fixture preparation, before recovery authority.
    actor, snapshot_a, snapshot_b, x = _fixtures(family)
    codec = CoupledCheckpointCodec()
    raw_a = codec.encode(snapshot_a, binding=BINDING, limits=LIMITS)
    raw_b = codec.encode(snapshot_b, binding=BINDING, limits=LIMITS)
    digest_a, digest_b = sha256(raw_a).hexdigest(), sha256(raw_b).hexdigest()
    path = directory / "checkpoint.json"
    path.parent.mkdir()
    store = BoundedCheckpointFileStore(path, codec)
    if store.save(snapshot_a, binding=BINDING, limits=LIMITS) != digest_a or raw_a == raw_b:
        raise AssertionError("native fixture A/B did not establish distinct checkpoints")
    predictions = {"before": actor.predict_proba(x.copy()).tolist()}
    authority: dict[str, Any] = {}
    with ExitStack() as stack:
        child_one = _new_child(stack)
        api = WindowsRecoveryApi()
        anchor = stack.enter_context(WindowsProcessHandle.pin(os.getpid(), api=api))
        first = stack.enter_context(WindowsProcessHandle.pin(child_one.pid, api=api))
        observer = WindowsRecoveryObserver(anchor)
        initial = observer.observe(first)
        known = _authority(initial, anchor.identity, first.identity, digest_a, family)
        journal = SqliteRecoveryJournal.create(directory / "authority.sqlite", known)
        coordinator = stack.enter_context(
            RecoveryCoordinator(known, journal, observer, anchor, first)
        )
        authority["initial"] = asdict(journal.read())

        def stage():
            current = journal.read()
            if current.metadata.usage.checkpoint_attempts != 1:
                raise AssertionError("checkpoint attempt was not charged before child work")
            child_one.send(_request(path, digest_b, payload=raw_b))
            return child_one.receive()

        staged = coordinator.execute(
            RecoveryCosts(copy_bytes=2 * LIMITS.max_encoded_bytes, checkpoint_attempts=1), stage
        )
        # Kill outside execute: its post-prepare checks require the writer alive.
        authority["reserved"] = asdict(journal.read())
        live = observer.observe(first)
        temporary = _staging_path(path, staged, raw_b)
        predictions["during_staging"] = actor.predict_proba(x.copy()).tolist()
        if live.previous_owner_ended is not False or _bytes(path) != raw_a:
            raise AssertionError("staging published B or writer was not alive")
        first_exit = child_one.kill()
        ended = observer.observe(first)
        if ended.previous_owner_ended is not True or _bytes(path) != raw_a:
            raise AssertionError("ended writer changed A or lacks retained ended proof")
        staged_raw = _bytes(_staging_path(path, staged, raw_b))
        temporary.unlink()  # Exact owned orphan, only after ended proof and byte verification.
        predictions["after_exit"] = actor.predict_proba(x.copy()).tolist()

        child_two = _new_child(stack)
        second = stack.enter_context(WindowsProcessHandle.pin(child_two.pid, api=api))
        coordinator.handoff(second, "writer-two")
        authority["handed_off"] = asdict(journal.read())

        def restore():
            child_two.send(_request(path, digest_a))
            return child_two.receive()

        recovered = coordinator.execute(
            RecoveryCosts(copy_bytes=2 * LIMITS.max_encoded_bytes), restore
        )
        authority["read_done"] = asdict(journal.read())
        if recovered != dict(
            event="restored", snapshot=_plain(asdict(snapshot_a)), predictions=predictions["before"]
        ):
            raise AssertionError("replacement worker did not recover complete A/predictions")
        predictions["after_handoff"] = actor.predict_proba(x.copy()).tolist()
        callbacks = []
        try:
            coordinator.execute(
                RecoveryCosts(checkpoint_attempts=1), lambda: callbacks.append(True)
            )
        except ValueError as error:
            refusal = str(error)
            if (
                refusal != "original authority allowance exceeded"
                or child_two.returncode is not None
            ):
                raise AssertionError(
                    "checkpoint request did not refuse at the live attempt cap"
                ) from error
        else:
            raise AssertionError("exhausted checkpoint attempts were renewed")
        terminal = journal.read()
        authority["terminal"] = asdict(terminal)
        if (
            callbacks
            or not terminal.metadata.stopped
            or terminal.metadata.usage.checkpoint_attempts != 2
        ):
            raise AssertionError("exhaustion did not retain spent attempts before callback")
        if terminal.metadata.sequence > 40:
            raise AssertionError("example exceeded its authority transition bound")
        predictions["after_refusal"] = actor.predict_proba(x.copy()).tolist()
        child_two.send({"op": "stop"})
        second_exit = child_two.wait()
        if second_exit != 0 or child_one.stderr or child_two.stderr:
            raise AssertionError("owned writer ended with unexpected diagnostics")
        child_pids = [child_one.pid, child_two.pid]
    if _bytes(path) != raw_a or temporary.exists():
        raise AssertionError("checkpoint A/orphan cleanup changed after handoff")
    if any(value != predictions["before"] for value in predictions.values()):
        raise AssertionError("writer interruption changed the serving actor predictions")
    np.testing.assert_equal(asdict(actor.snapshot_state()), asdict(snapshot_a))
    return dict(
        family=family,
        path=str(path),
        checkpoint_a=dict(
            digest=digest_a, payload_utf8=raw_a.decode(), snapshot=asdict(snapshot_a)
        ),
        candidate_b=dict(digest=digest_b, payload_utf8=raw_b.decode(), snapshot=asdict(snapshot_b)),
        predictions=predictions,
        staged=staged,
        staged_payload_utf8=staged_raw.decode(),
        staged_file_removed=True,
        recovered=recovered,
        authority=authority,
        observations=dict(initial=asdict(initial), staged=asdict(live), ended=asdict(ended)),
        child_pids=child_pids,
        child_exit_codes=[first_exit, second_exit],
        attempt_refusal=refusal,
        refused_callback_calls=len(callbacks),
    )


def run_example(directory: Path) -> dict[str, Any]:
    if sys.platform != "win32":
        raise OSError("supervised checkpoint interruption requires Windows process observation")
    directory = directory.absolute()
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    rows = [_family_example(directory / family, family) for family in FAMILIES]
    if time.monotonic() - started > 60:
        raise RuntimeError("checkpoint interruption example exceeded60s")
    for phase in ("checkpoint_a", "candidate_b"):
        np.testing.assert_equal(
            rows[0][phase]["snapshot"]["parameters"], rows[1][phase]["snapshot"]["parameters"]
        )
    remaining = [str(path) for path in directory.glob("*/.*.tmp")]
    retained = sum(path.stat().st_size for path in directory.glob("*/*") if path.is_file())
    if remaining or retained > 16 * 1024**2:
        raise AssertionError("example retained orphan files or exceeded16MiB")
    return _plain(
        dict(
            **describe_example(),
            directory=str(directory),
            rows=rows,
            remaining_temporary_files=remaining,
            retained_file_bytes=retained,
        )
    )
