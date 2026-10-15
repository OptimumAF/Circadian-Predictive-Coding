"""Validate trusted pause telemetry and stop exactly witnessed pending work.

Inputs are an independently retained clean authority/reference and a bounded
private-channel reply. Output is pending authority, never completion authority.
Terminal persistence delegates the existing exact-witness SQLite path; the fixed
outer control terminates only its owned peer and records genuine ended facts
separately. No native learning, disk-derived floor, retry or observation relabeling.
"""

from dataclasses import asdict, replace
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import Any, NoReturn

from src.adapters.native_transfer_protocol import BINDING, LIMITS
from src.adapters.coupled_checkpoint_codec import CoupledCheckpointCodec
from src.adapters.native_transfer_session import NativeTransferSession
from src.adapters.windows_transfer_example import _require
from src.core.coordinator_transfer import SupervisorRecoveryWitness
from src.core.coordinator_transfer import _authority
from src.core.recovery_authority import AuthorityRecord, validate_authority_floor
from src.core.recovery_reporting import FailureWitness, plan_report
from src.infra.recovery_authority_codec import decode_authority
from src.infra.sqlite_recovery_journal import SqliteRecoveryJournal
from src.infra.checkpoint_files import BoundedCheckpointFileStore
from src.infra.windows_recovery_observer import WindowsRecoveryObserver


def validate_pause(
    before: AuthorityRecord,
    response: dict[str, Any],
    expected_payload: bytes,
    expected_energy: float,
) -> AuthorityRecord:
    """Bind diagnostic C to the spent reservation while preserving completed B."""
    _authority(before)
    metadata = before.metadata
    if (
        metadata.stopped
        or metadata.uncertain_work
        or metadata.usage.updates_admitted != metadata.usage.updates_completed
        or metadata.manifest.native_sha256 != metadata.manifest.payload_sha256
        or metadata.manifest.source_sha256 != BINDING.source_sha256
        or metadata.manifest.policy_sha256 != BINDING.policy_sha256
    ):
        raise ValueError("native pause requires independently known clean native-only authority")
    if (
        type(response) is not dict
        or response.keys()
        != {"event", "authority_utf8", "payload_utf8", "digest", "energy", "budget"}
        or type(response["event"]) is not str
        or response["event"] != "native_paused"
        or any(
            type(response[name]) is not str for name in ("authority_utf8", "payload_utf8", "digest")
        )
    ):
        raise ValueError("native pause reply differs from exact diagnostic wire")
    budget = dict(
        constructions=1,
        train_calls=1,
        native_sgd=1,
        raw_scaling_sgd=0,
        energy=4,
        predictions=1,
        split=0,
        guarded_inference=0,
        popen=0,
    )
    if (
        type(response["budget"]) is not dict
        or response["budget"] != budget
        or any(type(value) is not int for value in response["budget"].values())
    ):
        raise ValueError("native pause counters differ from charged fixed work")
    if (
        type(expected_payload) is not bytes
        or not 0 < len(expected_payload) <= LIMITS.max_encoded_bytes
        or response["payload_utf8"].encode("utf-8") != expected_payload
        or response["digest"] != sha256(expected_payload).hexdigest()
        or type(expected_energy) not in (int, float)
        or type(response["energy"]) not in (int, float)
    ):
        raise ValueError("native pause diagnostics differ from independently expected C")
    try:
        finite = isfinite(expected_energy) and isfinite(response["energy"])
    except OverflowError:
        finite = False
    if not finite or response["energy"] != expected_energy:
        raise ValueError("native pause energy differs from finite independent reference")
    pending = decode_authority(response["authority_utf8"].encode("utf-8"))
    validate_authority_floor(before, pending)
    expected = replace(
        before,
        metadata=replace(
            metadata,
            sequence=metadata.sequence + 2,
            observed_ns=pending.metadata.observed_ns,
            uncertain_work=True,
            usage=replace(
                metadata.usage,
                updates_admitted=metadata.usage.updates_admitted + 1,
                copied_bytes=metadata.usage.copied_bytes + 4 * LIMITS.max_encoded_bytes,
                checkpoint_attempts=metadata.usage.checkpoint_attempts + 1,
                peak_rss_bytes=pending.metadata.usage.peak_rss_bytes,
            ),
        ),
    )
    if pending != expected:
        raise ValueError("native pause changed protected authority or original reservation charge")
    return pending


def stop_interrupted(path: Path, pending: AuthorityRecord) -> AuthorityRecord:
    """Stop at the child's last known stamp; supervisor facts stay separate."""
    _authority(pending)
    metadata = pending.metadata
    if (
        metadata.stopped
        or not metadata.uncertain_work
        or metadata.usage.updates_admitted != metadata.usage.updates_completed + 1
        or metadata.manifest.native_sha256 != metadata.manifest.payload_sha256
    ):
        raise ValueError("interrupted stop requires one independently witnessed pending update")
    expected = plan_report(pending, None, terminal=True).proposed
    actual = SqliteRecoveryJournal.reconcile_terminal(path, FailureWitness(pending))
    if actual != expected:
        raise ValueError("interrupted terminal persistence differs from exact last-known stop")
    return actual


def _refused(action, expected):
    try:
        action()
    except ValueError as error:
        _require(str(error) == expected, "native interruption refusal reason differs")
        return str(error)
    raise AssertionError("interrupted native work acquired unsupported authority")


class _ForbiddenPorts:
    """Refused adoption must precede every observer/registration callback."""

    @property
    def identity(self) -> NoReturn:
        raise AssertionError("interrupted adoption read a registration")

    def is_ended(self) -> NoReturn:
        raise AssertionError("interrupted adoption probed a process")

    def close(self) -> NoReturn:
        raise AssertionError("interrupted adoption closed a registration")

    def observe(self, previous_owner=None) -> NoReturn:
        raise AssertionError("interrupted adoption sampled a host")


def _interruption_refusals(path, pending, terminal, fixture, own, supervisor):
    def forbidden(*args, **kwargs):
        raise AssertionError("interrupted adoption reached a forbidden callback")

    refusals, ports = {}, _ForbiddenPorts()
    for name, known in (("pending", pending), ("terminal", terminal)):
        refusals[f"{name}_adoption"] = _refused(
            lambda: NativeTransferSession(
                known, path, ports, ports, ports, model_factory=forbidden
            ),
            "native transfer requires independently known clean native-only bindings",
        )
        refusals[f"{name}_transfer"] = _refused(
            lambda: SupervisorRecoveryWitness(known, own.identity),
            "transfer requires completed native-only authority and a distinct supervisor",
        )
    guessed = replace(
        terminal,
        metadata=replace(
            terminal.metadata,
            manifest=replace(
                terminal.metadata.manifest,
                native_sha256=fixture["digests"]["c"],
                payload_sha256=fixture["digests"]["c"],
            ),
        ),
    )
    refusals["guessed_c"] = _refused(
        lambda: SqliteRecoveryJournal(path, guessed).read(), "original authority manifest differs"
    )
    # The earlier transfer witness cannot erase a later admitted native update.
    try:
        SqliteRecoveryJournal.reconcile_coordinator_transfer(path, supervisor.failure_witness())
    except ValueError as error:
        _require(
            str(error) == "coordinator reconciliation requires exact witnessed disk state",
            "stale transfer refusal reason differs",
        )
        refusals["stale_transfer"] = str(error)
    else:
        raise AssertionError("stale transfer witness overwrote newer native work")
    _require(
        SqliteRecoveryJournal(path, terminal).read() == terminal,
        "interruption refusal changed persisted terminal authority",
    )
    return refusals


def interrupt_replacement(peer, paths, restored_b, fixture, path, own, anchor, worker, supervisor):
    peer.send(
        dict(
            op="update",
            source=str(paths["b"]),
            target=str(paths["c"]),
            expected_source_sha256=fixture["digests"]["b"],
            expected_target_sha256=fixture["digests"]["c"],
        )
    )
    paused = peer.receive()
    pending = validate_pause(restored_b, paused, fixture["payloads"]["c"], fixture["energies"]["c"])
    _require(
        SqliteRecoveryJournal(path, pending).read() == pending,
        "independently received pending authority differs from disk",
    )
    _require(
        CoupledCheckpointCodec().encode(
            fixture["actor"].snapshot_state(), binding=BINDING, limits=LIMITS
        )
        == fixture["payloads"]["a"],
        "replacement mutation changed actor full A",
    )
    _require(not paths["c"].exists(), "interrupted replacement saved C before pause")
    observer = WindowsRecoveryObserver(own, peak_rss_bytes=pending.metadata.usage.peak_rss_bytes)
    live = observer.observe(anchor)
    _require(live.previous_owner_ended is False, "replacement ended before controlled kill")
    killed = peer.kill()
    ended = observer.observe(anchor)
    _require(
        type(killed) is int
        and killed != 0
        and peer.returncode == killed
        and peer.stderr == b""
        and anchor.is_ended()
        and ended.previous_owner_ended is True
        and worker.is_ended() is False,
        "replacement interruption did not reap the owned coordinator with live worker",
    )
    terminal = stop_interrupted(path, pending)
    refusals = _interruption_refusals(path, pending, terminal, fixture, own, supervisor)
    codec = CoupledCheckpointCodec()
    for name in ("a", "b"):
        snapshot = BoundedCheckpointFileStore(paths[name], codec).load(
            binding=BINDING, expected_sha256=fixture["digests"][name], limits=LIMITS
        )
        _require(
            codec.encode(snapshot, binding=BINDING, limits=LIMITS) == fixture["payloads"][name],
            "replacement interruption changed independently completed A/B file",
        )
    _require(not paths["c"].exists(), "replacement interruption unexpectedly persisted C")
    return (
        dict(
            paused=paused,
            pending=asdict(pending),
            terminal=asdict(terminal),
            refusals=refusals,
            observations=dict(replacement_live=asdict(live), replacement_ended=asdict(ended)),
            candidate_not_persisted=True,
        ),
        terminal,
    )
