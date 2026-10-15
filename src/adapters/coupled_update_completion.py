"""Native-only completion through the existing surviving Windows coordinator.

Reference native fixtures precede authority creation; each child update is then
admitted and charged before execution. Parent validation owns checkpoint B, while
the actor and checkpoint A remain unchanged. Full managed-bundle promotion and
coordinator-loss recovery are not provided.
Opt-in owned-file corruption verifies that damaged candidate bytes cannot confer
completion, while retaining the original expected digest and forensic evidence.
"""

from contextlib import ExitStack
from dataclasses import asdict, replace
from hashlib import sha256
import os
from pathlib import Path
import sys
import sysconfig
import time
from typing import Any

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np

from src.adapters.coupled_checkpoint_codec import CoupledCheckpointCodec
from src.adapters.coupled_checkpoint_interruption import FEATURES, LIMITS, _bytes, _model, _plain
from src.app.recovery_coordinator import RecoveryCoordinator
from src.core.checkpoint_codec import CodecBinding
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

CASES = ("pc", "neutral_cpc", "active_cpc", "active_cpc_validation_failure")
BINDING = CodecBinding(
    sha256(b"coupled completion engineering source v1").hexdigest(),
    sha256(b"coupled completion engineering policy v1").hexdigest(),
)


def _corruption_option(value: bool) -> None:
    if type(value) is not bool:
        raise ValueError("corrupt_checkpoint must be an exact boolean")


def describe_example(*, corrupt_checkpoint: bool = False) -> dict[str, Any]:
    _corruption_option(corrupt_checkpoint)
    cases = CASES[:3] if corrupt_checkpoint else CASES
    report = dict(
        protocol_id="coupled_checkpoint_corruption_engineering_v1"
        if corrupt_checkpoint
        else "coupled_update_completion_engineering_v1",
        cases=list(cases),
        native_training_calls=4 * len(cases),
        native_energy_evaluations=16 * len(cases),
        owned_child_launches=len(cases),
        preparatory_training_calls=3 * len(cases),
        admitted_child_updates=len(cases),
        seed=23,
        geometry=dict(input_dim=2, hidden_dims=[3, 2], precisions=[1.0, 1.0]),
        training=dict(
            role="train", features=[list(row) for row in FEATURES], targets=[[1.0], [0.0]]
        ),
        external_training_settings=dict(
            learning_rate=0.02, inference_steps=2, inference_learning_rate=0.05
        ),
        limits=asdict(LIMITS),
        binding=asdict(BINDING),
        limitations="Native-only candidate completion, surviving Windows parent. Actor A unchanged. No full bundle promotion, uncertain retry or coordinator-loss recovery; fixture bindings are not authentication.",
    )
    if corrupt_checkpoint:
        report["native_prediction_calls"] = 21
        report["corruption_mode"] = (
            "Quiescent owned candidate file only; final byte125 becomes33 before load using original expected digest"
        )
    return report


def _corrupt_checkpoint(path: Path, valid_payload: bytes, expected_digest: str) -> dict[str, Any]:
    """Inject one explicit byte fault only into this fixture's owned candidate.

    This trusted quiescent-path helper is not a hostile-filesystem operation or
    an arbitrary-file CLI. Keep damaged bytes as evidence; never bless their hash.
    """
    if type(valid_payload) is not bytes or not valid_payload or valid_payload[-1:] != b"}":
        raise ValueError("corruption fixture requires exact nonempty JSON bytes ending in }")
    if sha256(valid_payload).hexdigest() != expected_digest:
        raise ValueError("corruption fixture original digest differs from valid payload")
    valid_payload.decode("utf-8")
    if _bytes(path) != valid_payload:
        raise ValueError("corruption fixture owned file differs from valid payload")
    offset = len(valid_payload) - 1
    with path.open("r+b") as stream:
        stream.seek(offset)
        count = stream.write(b"!")
        if type(count) is not int or count != 1:
            raise OSError("corruption fixture did not write exactly one byte")
        stream.flush()
        os.fsync(stream.fileno())
    damaged = _bytes(path)
    if damaged != valid_payload[:-1] + b"!":
        raise AssertionError("corruption fixture changed bytes beyond its final-byte fault")
    return dict(
        offset=offset,
        original_byte=125,
        replacement_byte=33,
        expected_digest=expected_digest,
        actual_digest=sha256(damaged).hexdigest(),
        payload_utf8=damaged.decode("utf-8"),
    )


def _fixtures(family):
    actor = _model(family)
    x, y = np.asarray(FEATURES, dtype=np.float64), np.array([[1.0], [0.0]])
    for _ in range(2):
        actor.train_epoch(x.copy(), y.copy(), 0.02, 2, 0.05)
    a = actor.snapshot_state()
    reference = _model(family)
    reference.restore_state(a)
    energy = reference.train_epoch(x.copy(), y.copy(), 0.02, 2, 0.05).energy
    return actor, a, reference.snapshot_state(), energy, reference.predict_proba(x).tolist(), x


def _child(stack: ExitStack) -> BoundedJsonChild:
    executable = getattr(sys, "_base_executable", None)
    if type(executable) is not str or not Path(executable).is_file():
        raise ValueError("native completion requires the existing physical base interpreter")
    repository = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(repository), sysconfig.get_path("purelib")))
    child = stack.enter_context(
        BoundedJsonChild(
            [executable, "-u", "-B", "-X", "utf8", "-m", "src.adapters.coupled_update_worker"],
            cwd=repository,
            env=env,
        )
    )
    if child.receive() != dict(event="hello", pid=child.pid, version=list(sys.version_info[:3])):
        raise ValueError("native completion hello differs from the owned physical worker")
    return child


def _known(observation, anchor, worker, digest, case):
    absent = sha256(b"absent component in native-only completion fixture").hexdigest()
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
        f"native-completion-{case}-{anchor.pid}",
        observation.clock_epoch,
        "native-worker",
        0,
        0,
        observation.now_ns,
        observation.now_ns,
        0,
        manifest,
        RecoveryLimits(1, 60_000_000_000, 4 * LIMITS.max_encoded_bytes, 0, 1, 512 * 1024**2),
        RecoveryUsage(0, 0, 0, 0, 0, observation.peak_rss_bytes),
        False,
        False,
    )
    return AuthorityRecord(metadata, anchor, worker)


def _case(directory: Path, case: str, *, corrupt_checkpoint: bool = False) -> dict[str, Any]:
    _corruption_option(corrupt_checkpoint)
    validation_failure = case.endswith("_validation_failure")
    failed = validation_failure or corrupt_checkpoint
    family = "active_cpc" if validation_failure else case
    actor, a, b, expected_energy, expected_predictions, x = _fixtures(family)
    codec = CoupledCheckpointCodec()
    raw_a = codec.encode(a, binding=BINDING, limits=LIMITS)
    raw_b = codec.encode(b, binding=BINDING, limits=LIMITS)
    digest_a, digest_b = sha256(raw_a).hexdigest(), sha256(raw_b).hexdigest()
    directory.mkdir()
    path_a, path_b = directory / "actor-a.json", directory / "candidate-b.json"
    BoundedCheckpointFileStore(path_a, codec).save(a, binding=BINDING, limits=LIMITS)
    predictions = {"before": actor.predict_proba(x.copy()).tolist()}
    callbacks = dict(prepare=0, validate=0, probe=0, refused_prepare=0, refused_validate=0)
    records: dict[str, Any] = {}
    responses = []
    pending_states = []
    actual_b = None
    corruption = None
    with ExitStack() as stack:
        child = _child(stack)
        api = WindowsRecoveryApi()
        anchor = stack.enter_context(WindowsProcessHandle.pin(os.getpid(), api=api))
        worker = stack.enter_context(WindowsProcessHandle.pin(child.pid, api=api))
        observer = WindowsRecoveryObserver(anchor)
        initial = observer.observe(worker)
        known = _known(initial, anchor.identity, worker.identity, digest_a, case)
        journal = SqliteRecoveryJournal.create(directory / "authority.sqlite", known)
        coordinator = stack.enter_context(
            RecoveryCoordinator(known, journal, observer, anchor, worker)
        )
        records["initial"] = asdict(journal.read())

        def prepare():
            callbacks["prepare"] += 1
            pending = journal.read()
            if not pending.metadata.uncertain_work or pending.metadata.usage.updates_admitted != 1:
                raise AssertionError("native update was not charged before child dispatch")
            records["pending"] = asdict(pending)
            pending_states.append(pending)
            child.send(
                dict(
                    op="update",
                    binding=asdict(BINDING),
                    limits=asdict(LIMITS),
                    expected_sha256=digest_a,
                    payload_utf8=raw_a.decode(),
                )
            )
            predictions["pending"] = actor.predict_proba(x.copy()).tolist()
            response = child.receive()
            responses.append(response)
            return response

        def validate(response):
            nonlocal actual_b, corruption
            callbacks["validate"] += 1
            if set(response) != {
                "event",
                "payload_utf8",
                "digest",
                "energy",
                "predictions",
                "training_calls",
            }:
                raise ValueError("native worker response schema differs")
            if (
                response["event"] != "updated"
                or type(response["training_calls"]) is not int
                or response["training_calls"] != 1
                or response["energy"] != expected_energy
                or response["predictions"] != expected_predictions
                or response["digest"] != digest_b
                or type(response["payload_utf8"]) is not str
                or response["payload_utf8"].encode() != raw_b
            ):
                raise ValueError("native worker did not return the exact one-update reference B")
            decoded = codec.decode(raw_b, binding=BINDING, expected_sha256=digest_b, limits=LIMITS)
            np.testing.assert_equal(asdict(decoded), asdict(b))
            store_b = BoundedCheckpointFileStore(path_b, codec)
            if (
                store_b.save(decoded, binding=BINDING, limits=LIMITS) != digest_b
                or _bytes(path_b) != raw_b
            ):
                raise AssertionError(
                    "parent-owned B checkpoint differs from validated native bytes"
                )
            if corrupt_checkpoint:
                records["checkpoint_written"] = asdict(journal.read())
                predictions["checkpoint_written"] = actor.predict_proba(x.copy()).tolist()
                corruption = _corrupt_checkpoint(path_b, raw_b, digest_b)
            loaded = store_b.load(binding=BINDING, expected_sha256=digest_b, limits=LIMITS)
            if corrupt_checkpoint:
                raise AssertionError("corrupted candidate checkpoint unexpectedly reloaded")
            np.testing.assert_equal(asdict(loaded), asdict(b))
            actual_b = dict(digest=digest_b, payload_utf8=raw_b.decode(), snapshot=asdict(loaded))
            records["validated"] = asdict(journal.read())
            predictions["validated"] = actor.predict_proba(x.copy()).tolist()
            if failed:
                raise ValueError("injected validator refusal after valid B checkpoint write")
            return replace(known.metadata.manifest, payload_sha256=digest_b, native_sha256=digest_b)

        native_failure = None
        try:
            response = coordinator.execute_native_update(
                RecoveryCosts(
                    updates=1, copy_bytes=4 * LIMITS.max_encoded_bytes, checkpoint_attempts=1
                ),
                prepare,
                validate,
            )
            if failed or response != responses[0]:
                raise AssertionError("native failure fixture was accepted or returned wrong result")
        except ValueError as error:
            expected_failure = (
                f"checkpoint content differs from independently expected digest: {path_b}"
                if corrupt_checkpoint
                else "injected validator refusal after valid B checkpoint write"
            )
            if (
                not failed
                or str(error) != expected_failure
                or (corrupt_checkpoint and type(error) is not ValueError)
            ):
                raise
            native_failure = str(error)
        records["result"] = asdict(journal.read())
        predictions["after_result"] = actor.predict_proba(x.copy()).tolist()
        after_result = observer.observe(worker)
        stale_floor_refusal = None
        records["probe"] = None
        if not failed:

            def probe(value):
                callbacks["probe"] += 1
                if value != digest_b:
                    raise AssertionError("publication probe lost completed candidate identity")

            coordinator.execute(RecoveryCosts(), lambda: digest_b, probe)
            records["probe"] = asdict(journal.read())
            # Use the original live pending record, never authority reconstructed
            # from a checkpoint's or a report's self-described JSON fields.
            stale = SqliteRecoveryJournal(journal.path, pending_states[0])
            try:
                stale.read()
            except ValueError as error:
                stale_floor_refusal = str(error)
            else:
                raise AssertionError(
                    "stale pending floor accepted self-asserted completion from disk"
                )

        def refused_prepare():
            callbacks["refused_prepare"] += 1
            raise AssertionError("exhausted/failed native update dispatched again")

        def refused_validate(value):
            callbacks["refused_validate"] += 1
            raise AssertionError("exhausted/failed native validator ran again")

        try:
            coordinator.execute_native_update(
                RecoveryCosts(updates=1), refused_prepare, refused_validate
            )
        except ValueError as error:
            final_refusal = str(error)
            expected = (
                "coordinator previously failed; independent reconciliation required"
                if failed
                else "original authority allowance exceeded"
            )
            if final_refusal != expected or child.returncode is not None:
                raise AssertionError(
                    "native retry did not refuse at the expected live guard"
                ) from error
        else:
            raise AssertionError("native update allowance was renewed")
        records["terminal"] = asdict(journal.read())
        if corrupt_checkpoint:
            terminal = journal.read().metadata
            if (
                not terminal.stopped
                or not terminal.uncertain_work
                or terminal.manifest != known.metadata.manifest
                or terminal.usage.updates_admitted != 1
                or terminal.usage.updates_completed != 0
                or terminal.usage.copied_bytes != 4 * LIMITS.max_encoded_bytes
                or terminal.usage.checkpoint_attempts != 1
                or terminal.usage.grants != 0
                or callbacks
                != dict(prepare=1, validate=1, probe=0, refused_prepare=0, refused_validate=0)
                or actual_b is not None
                or corruption is None
            ):
                raise AssertionError(
                    "corrupted checkpoint changed completion, bindings or spent costs"
                )
        if records["terminal"]["metadata"]["sequence"] > 64:
            raise AssertionError("native example exceeded its authority transition bound")
        after_refusal = observer.observe(worker)
        predictions["after_refusal"] = actor.predict_proba(x.copy()).tolist()
        child.send({"op": "stop"})
        exit_code = child.wait()
        if exit_code != 0 or child.stderr:
            raise AssertionError("native worker did not stop cleanly")
        identity = asdict(worker.identity)
        child_pid = child.pid
    if _bytes(path_a) != raw_a or any(p != predictions["before"] for p in predictions.values()):
        raise AssertionError("native candidate completion changed actor A")
    np.testing.assert_equal(asdict(actor.snapshot_state()), asdict(a))
    row = dict(
        case=case,
        family=family,
        expected_failure=failed,
        path_a=str(path_a),
        path_b=str(path_b),
        checkpoint_a=dict(digest=digest_a, payload_utf8=raw_a.decode(), snapshot=asdict(a)),
        expected_b=dict(digest=digest_b, payload_utf8=raw_b.decode(), snapshot=asdict(b)),
        actual_b=actual_b,
        response=responses[0],
        predictions=predictions,
        expected_b_predictions=expected_predictions,
        authority=records,
        callbacks=callbacks,
        native_failure=native_failure,
        final_attempt_refusal=final_refusal,
        stale_floor_refusal=stale_floor_refusal,
        worker_identity=identity,
        observations=dict(
            initial=asdict(initial),
            after_result=asdict(after_result),
            after_refusal=asdict(after_refusal),
        ),
        child_pid=child_pid,
        exit_code=exit_code,
    )
    if corrupt_checkpoint:
        row["corruption"] = corruption
    return row


def run_example(directory: Path, *, corrupt_checkpoint: bool = False) -> dict[str, Any]:
    _corruption_option(corrupt_checkpoint)
    if sys.platform != "win32":
        raise OSError("native completion example requires Windows retained-process observation")
    directory = directory.absolute()
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    rows = (
        [_case(directory / case, case, corrupt_checkpoint=True) for case in CASES[:3]]
        if corrupt_checkpoint
        else [_case(directory / case, case) for case in CASES]
    )
    if time.monotonic() - started > 60:
        raise RuntimeError("native completion example exceeded60s")
    phases = (
        ("checkpoint_a", "expected_b")
        if corrupt_checkpoint
        else ("checkpoint_a", "expected_b", "actual_b")
    )
    for phase in phases:
        np.testing.assert_equal(
            rows[0][phase]["snapshot"]["parameters"], rows[1][phase]["snapshot"]["parameters"]
        )
    remaining = [str(path) for path in directory.glob("*/.*.tmp")]
    retained = sum(path.stat().st_size for path in directory.glob("*/*") if path.is_file())
    if remaining or retained > 64 * 1024**2:
        raise AssertionError("native completion retained an orphan or exceeded storage bounds")
    return _plain(
        dict(
            **(
                describe_example(corrupt_checkpoint=True)
                if corrupt_checkpoint
                else describe_example()
            ),
            directory=str(directory),
            rows=rows,
            remaining_temporary_files=remaining,
            retained_file_bytes=retained,
        )
    )
