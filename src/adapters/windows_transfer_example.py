"""One physical coordinator replacement using a zero-native authority fixture.

Composes retained Windows supervisor, owned fixed child protocol and private
SQLite. Demonstrates metadata authority continuation and original quota stopping;
does not restore a native checkpoint, train, or claim complete crash recovery.
"""

from contextlib import ExitStack
from dataclasses import asdict, replace
from hashlib import sha256
import os
from pathlib import Path
import sys
import sysconfig
from typing import Any

from src.core.recovery_admission import (
    RecoveryLimits,
    RecoveryManifest,
    RecoveryMetadata,
    RecoveryUsage,
)
from src.core.recovery_authority import AuthorityRecord, validate_authority_floor
from src.core.recovery_observation import (
    RecoveryHostObservation,
    RecoveryProcessIdentity,
    anchored_clock_epoch,
)
from src.infra.bounded_json_child import BoundedJsonChild
from src.infra.recovery_authority_codec import decode_authority, encode_authority
from src.infra.sqlite_recovery_journal import SqliteRecoveryJournal
from src.infra.windows_process_handles import WindowsProcessHandle, WindowsRecoveryApi
from src.infra.windows_transfer_supervisor import WindowsTransferSupervisor


def describe_example() -> dict[str, Any]:
    return dict(
        protocol_id="windows_coordinator_transfer_engineering_v1",
        owned_children=4,
        native_work=0,
        checkpoint_restoration=False,
        max_elapsed_seconds=60,
        max_rss_bytes=512 * 1024**2,
        max_checkpoint_attempts=2,
        max_copy_bytes=10,
        limits="Windows only; fixed trusted children and synthetic zero-native manifest. Physical process identity/clock/SQLite continuation, not full native recovery or comparative learning evidence.",
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _child(stack: ExitStack) -> BoundedJsonChild:
    executable = getattr(sys, "_base_executable", None)
    if type(executable) is not str or not Path(executable).is_file():
        raise ValueError("existing physical base interpreter required")
    repository = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    env["PYTHONPATH"] = os.pathsep.join((str(repository), sysconfig.get_path("purelib")))
    peer = stack.enter_context(
        BoundedJsonChild(
            [executable, "-u", "-B", "-X", "utf8", "-m", "src.adapters.windows_transfer_worker"],
            cwd=repository,
            env=env,
            timeout_seconds=5,
            max_line_bytes=65536,
            max_output_bytes=262144,
            max_stderr_bytes=16384,
            max_messages=8,
        )
    )
    _require(
        peer.receive() == dict(event="hello", pid=peer.pid, version=list(sys.version_info[:3])),
        "child hello differs from owned physical PID/interpreter",
    )
    return peer


def _initial_observation(peer, worker, api):
    before = api.interrupt_ns()
    seed = api.rss_bytes()
    _require(type(seed) is int and seed > 0, "supervisor RSS unavailable")
    peer.send(dict(op="observe", worker=asdict(worker.identity), peak_rss_bytes=seed))
    response = peer.receive()
    after = api.interrupt_ns()
    _require(
        set(response) == {"event", "observation"} and response["event"] == "observed",
        "initial native observation response differs",
    )
    value = dict(response["observation"])
    value["observer"] = RecoveryProcessIdentity(**value["observer"])
    value["previous_owner"] = RecoveryProcessIdentity(**value["previous_owner"])
    observation = RecoveryHostObservation(**value)
    _require(
        before <= observation.now_ns <= after
        and observation.previous_owner == worker.identity
        and observation.previous_owner_ended is False,
        "initial native observation not bound/live/in clock bracket",
    )
    return observation


def _initial_authority(observation, anchor, worker) -> AuthorityRecord:
    digest = sha256(
        b"zero-native physical authority engineering fixture; no native payload"
    ).hexdigest()
    manifest = RecoveryManifest(*([digest] * 9))
    return AuthorityRecord(
        RecoveryMetadata(
            1,
            f"physical-transfer-{anchor.pid}",
            observation.clock_epoch,
            "owner-original",
            0,
            0,
            observation.now_ns,
            observation.now_ns,
            0,
            manifest,
            RecoveryLimits(0, 60_000_000_000, 10, 0, 2, 512 * 1024**2),
            RecoveryUsage(0, 0, 0, 0, 0, observation.peak_rss_bytes),
            False,
            False,
        ),
        anchor,
        worker,
    )


def _bind_original(peer, path, original):
    peer.send(
        dict(
            op="bind",
            authority_utf8=encode_authority(original).decode("utf-8"),
            journal_path=str(path),
        )
    )
    response = peer.receive()
    _require(
        set(response) == {"event", "authority_utf8", "callbacks", "native_calls"}
        and response["event"] == "bound",
        "original coordinator bind response differs",
    )
    known = decode_authority(response["authority_utf8"].encode("utf-8"))
    validate_authority_floor(original, known)
    _require(
        known.anchor == original.anchor and known.worker == original.worker,
        "original bound registrations changed",
    )
    _require(
        known.metadata
        == replace(
            original.metadata,
            sequence=known.metadata.sequence,
            observed_ns=known.metadata.observed_ns,
            usage=replace(
                original.metadata.usage, peak_rss_bytes=known.metadata.usage.peak_rss_bytes
            ),
        ),
        "original binding changed protected authority or charged work",
    )
    _require(
        known.metadata.sequence > original.metadata.sequence,
        "original bound coordinator did not advance",
    )
    _require(
        response["callbacks"] == ["bound"]
        and type(response["native_calls"]) is int
        and response["native_calls"] == 0,
        "original bound callbacks/native work differ",
    )
    _require(
        SqliteRecoveryJournal(path, known).read() == known, "original binding not persisted exactly"
    )
    return known


def _continued(peer, path, change):
    peer.send(
        dict(
            op="continue",
            authority_utf8=encode_authority(change.proposed).decode("utf-8"),
            journal_path=str(path),
        )
    )
    response = peer.receive()
    _require(
        set(response) == {"event", "accepted_utf8", "terminal_utf8", "callbacks", "native_calls"}
        and response["event"] == "continued",
        "continuation response fields differ",
    )
    accepted = decode_authority(response["accepted_utf8"].encode("utf-8"))
    terminal = decode_authority(response["terminal_utf8"].encode("utf-8"))
    proposed = change.proposed
    validate_authority_floor(proposed, accepted)
    validate_authority_floor(accepted, terminal)
    _require(
        accepted.metadata
        == replace(
            proposed.metadata,
            sequence=accepted.metadata.sequence,
            observed_ns=accepted.metadata.observed_ns,
            usage=accepted.metadata.usage,
        )
        and accepted.metadata.sequence > proposed.metadata.sequence,
        "continued authority changed a protected field or failed to advance",
    )
    _require(
        terminal.metadata
        == replace(
            accepted.metadata,
            sequence=terminal.metadata.sequence,
            observed_ns=terminal.metadata.observed_ns,
            usage=terminal.metadata.usage,
            stopped=True,
        )
        and terminal.metadata.sequence > accepted.metadata.sequence,
        "terminal authority changed a protected field or failed to advance",
    )
    _require(
        accepted.anchor == proposed.anchor and accepted.worker == proposed.worker,
        "continued process identities changed",
    )
    _require(
        accepted.metadata.usage
        == replace(
            proposed.metadata.usage,
            copied_bytes=1,
            checkpoint_attempts=2,
            peak_rss_bytes=accepted.metadata.usage.peak_rss_bytes,
        ),
        "continued original cumulative costs differ",
    )
    _require(
        not accepted.metadata.stopped and not accepted.metadata.uncertain_work,
        "accepted continuation is not clean",
    )
    _require(
        terminal.metadata.stopped and not terminal.metadata.uncertain_work,
        "exhausted continuation must stop",
    )
    _require(
        terminal.metadata.usage
        == replace(accepted.metadata.usage, peak_rss_bytes=terminal.metadata.usage.peak_rss_bytes),
        "stopped continuation refunded or recharged work",
    )
    _require(
        terminal.metadata.observed_ns - terminal.metadata.started_ns
        < terminal.metadata.limits.max_elapsed_ns,
        "physical control exceeded original elapsed allowance",
    )
    _require(
        max(accepted.metadata.usage.peak_rss_bytes, terminal.metadata.usage.peak_rss_bytes)
        <= proposed.metadata.limits.max_rss_bytes,
        "physical control exceeded original RSS allowance",
    )
    _require(
        response["callbacks"] == ["prepare", "published"]
        and type(response["native_calls"]) is int
        and response["native_calls"] == 0,
        "unexpected callbacks/native work",
    )
    _require(
        SqliteRecoveryJournal(path, terminal).read() == terminal, "persisted terminal state differs"
    )
    return accepted, terminal, response["callbacks"]


def _crash_original(peer) -> None:
    """Accept only the fixed owned crash while preserving transport strictness."""
    peer.send(dict(op="crash"))
    try:
        peer.wait()
    except RuntimeError as error:
        if (
            str(error) != "child exit failed: returncode=17, stderr=b''"
            or type(peer.returncode) is not int
            or peer.returncode != 17
            or peer.stderr != b""
        ):
            raise
        # wait reports a nonzero exit even after successful cleanup. A separate
        # explicit close must succeed; an uncertain cleanup is never accepted.
        peer.close()
    else:
        raise ValueError("bounded transport did not refuse the original crash")


def run_example(directory: Path) -> dict[str, Any]:
    if sys.platform != "win32":
        raise OSError("physical transfer example requires Windows")
    directory = Path(directory).resolve()
    _require(directory.is_dir(), "example requires an existing private output directory")
    path = directory / "authority.sqlite"
    api = WindowsRecoveryApi()
    with ExitStack() as stack:
        own = stack.enter_context(WindowsProcessHandle.pin(api.current_pid(), api=api))
        old_peer, old_worker_peer = _child(stack), _child(stack)
        old_anchor = stack.enter_context(WindowsProcessHandle.pin(old_peer.pid, api=api))
        old_worker = stack.enter_context(WindowsProcessHandle.pin(old_worker_peer.pid, api=api))
        original_observation = _initial_observation(old_peer, old_worker, api)
        _require(
            original_observation.observer == old_anchor.identity
            and original_observation.clock_epoch == anchored_clock_epoch(old_anchor.identity),
            "initial observation does not identify physical old coordinator",
        )
        original = _initial_authority(
            original_observation, old_anchor.identity, old_worker.identity
        )
        journal = SqliteRecoveryJournal.create(path, original)
        original = _bind_original(old_peer, path, original)
        _require(
            journal.read() == original, "supervisor original floor differs from trusted bound reply"
        )
        supervisor = stack.enter_context(
            WindowsTransferSupervisor(original, own, old_anchor, old_worker)
        )
        _crash_original(old_peer)
        old_worker_peer.send(dict(op="exit"))
        _require(
            old_worker_peer.wait() == 0 and old_worker_peer.stderr == b"",
            "old owned worker did not exit cleanly",
        )
        _require(
            old_anchor.is_ended() and old_worker.is_ended(), "old retained processes have not ended"
        )
        new_peer, new_worker_peer = _child(stack), _child(stack)
        new_anchor = stack.enter_context(WindowsProcessHandle.pin(new_peer.pid, api=api))
        new_worker = stack.enter_context(WindowsProcessHandle.pin(new_worker_peer.pid, api=api))
        change = supervisor.plan_transfer(new_peer, new_anchor, new_worker, "owner-replacement")
        try:
            _require(
                supervisor.commit_transfer(journal) == change, "transfer acknowledgment differs"
            )
            accepted, terminal, callbacks = _continued(new_peer, path, change)
        except BaseException as error:
            try:
                SqliteRecoveryJournal.reconcile_coordinator_transfer(
                    path, supervisor.failure_witness()
                )
            except BaseException as reconciliation:
                error.add_note(
                    f"terminal transfer reconciliation refused/failed: {reconciliation!r}"
                )
            raise
        # The stale floor is deliberately independent; it cannot learn the new anchor.
        try:
            SqliteRecoveryJournal(path, original).read()
        except ValueError:
            pass
        else:
            raise ValueError("stale original floor accepted replacement anchor")
        for peer in (new_peer, new_worker_peer):
            peer.send(dict(op="exit"))
            _require(
                peer.wait() == 0 and peer.stderr == b"",
                "replacement owned process did not exit cleanly",
            )
        # Coordinator.close closed its local registrations, while these independent
        # retained references still prove both physical replacement processes ended.
        _require(
            new_anchor.is_ended() and new_worker.is_ended(),
            "replacement retained processes have not ended",
        )
        return dict(
            protocol=describe_example(),
            original=asdict(original),
            transfer=asdict(change),
            accepted=asdict(accepted),
            terminal=asdict(terminal),
            callbacks=callbacks,
            owned_returncodes=[
                peer.returncode for peer in (old_peer, old_worker_peer, new_peer, new_worker_peer)
            ],
            actual_owned_children=4,
            original_coordinator_bound=True,
            original_coordinator_crash_code=17,
            native_calls=0,
            physical_process_boundary_verified=True,
            native_checkpoint_recovery_verified=False,
        )
