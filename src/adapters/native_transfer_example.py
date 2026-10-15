"""Completed native B survives physical coordinator loss and continues to C.

Fixed PC/neutral/active engineering controls, trusted local files and owned
children. Full-state equality to uninterrupted references is checked; there is
no comparative learning, full managed-bundle, power-loss or lifecycle claim.
"""

from contextlib import ExitStack
from dataclasses import asdict, replace
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import sysconfig
from typing import Any

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np

from src.adapters.coupled_checkpoint_codec import CoupledCheckpointCodec
from src.adapters.coupled_checkpoint_interruption import _model
from src.adapters.native_transfer_protocol import (
    BINDING,
    LIMITS,
    FEATURES,
    FAMILIES,
    completion_from_wire,
)
from src.adapters.native_transfer_interruption import interrupt_replacement
from src.adapters.windows_transfer_example import _require, _initial_observation, _crash_original
from src.core.recovery_admission import (
    RecoveryLimits,
    RecoveryManifest,
    RecoveryMetadata,
    RecoveryUsage,
)
from src.core.recovery_authority import AuthorityRecord, validate_authority_floor
from src.infra.bounded_json_child import BoundedJsonChild
from src.infra.checkpoint_files import BoundedCheckpointFileStore
from src.infra.recovery_authority_codec import decode_authority, encode_authority
from src.infra.sqlite_recovery_journal import SqliteRecoveryJournal
from src.infra.windows_process_handles import WindowsProcessHandle, WindowsRecoveryApi
from src.infra.windows_transfer_supervisor import WindowsTransferSupervisor


def describe_example(*, interrupt_after_train: bool = False) -> dict[str, Any]:
    result = dict(
        protocol_id="native_coordinator_transfer_engineering_v1",
        families=list(FAMILIES),
        seed=23,
        geometry=dict(input_dim=2, hidden_dims=[3, 2], precisions=[1.0, 1.0]),
        training=dict(
            features=FEATURES,
            targets=[[1.0], [0.0]],
            role="train",
            learning_rate=0.02,
            inference_steps=2,
            inference_learning_rate=0.05,
        ),
        native_training_calls=18,
        native_energy_evaluations=72,
        owned_child_launches=12,
        original_limits=dict(
            updates=2,
            checkpoint_attempts=4,
            copy_bytes=10 * LIMITS.max_encoded_bytes,
            elapsed_seconds=60,
            rss_bytes=512 * 1024**2,
        ),
        checkpoint_limits=asdict(LIMITS),
        binding=asdict(BINDING),
        limitations="Windows, trusted fixed children/local files and continuously surviving supervisor. Complete raw native state only; no full managed lifecycle/actor promotion or scientific benefit claim.",
    )
    if interrupt_after_train:
        result.update(
            protocol_id="native_transfer_interruption_engineering_v1",
            interruption="Replacement terminated after native mutation, before session checkpoint save/completion. C telemetry is diagnostic only; completed B and spent uncertain reservation remain.",
        )
    return result


def _fixtures(family):
    codec = CoupledCheckpointCodec()
    x, y = np.array(FEATURES, dtype=np.float64), np.array([[1.0], [0.0]], dtype=np.float64)
    actor = _model(family)
    for _ in range(2):
        actor.train_epoch(x.copy(), y.copy(), 0.02, 2, 0.05)
    a = actor.snapshot_state()
    reference = _model(family)
    reference.restore_state(a)
    energy_b = reference.train_epoch(x.copy(), y.copy(), 0.02, 2, 0.05).energy
    b, predictions_b = reference.snapshot_state(), reference.predict_proba(x.copy()).tolist()
    energy_c = reference.train_epoch(x.copy(), y.copy(), 0.02, 2, 0.05).energy
    c, predictions_c = reference.snapshot_state(), reference.predict_proba(x.copy()).tolist()
    states = dict(a=a, b=b, c=c)
    payloads = {
        name: codec.encode(state, binding=BINDING, limits=LIMITS) for name, state in states.items()
    }
    if family != "pc":
        for expected_wakes, state in zip((2, 3, 4), states.values(), strict=True):
            wake = state.wake_state
            if wake is None:
                raise ValueError("CPC wake state is missing")
            _require(wake.wake_updates == expected_wakes, "CPC wake history was not retained")
            _require(np.any(wake.chemical_state != 0).item(), "CPC chemistry must be genuine")
    return dict(
        actor=actor,
        states=states,
        payloads=payloads,
        digests={name: sha256(raw).hexdigest() for name, raw in payloads.items()},
        energies=dict(b=energy_b, c=energy_c),
        predictions=dict(b=predictions_b, c=predictions_c),
        actor_prediction=actor.predict_proba(x.copy()).tolist(),
        x=x,
    )


def _child(stack, role, *, pause_after_train=False):
    executable = getattr(sys, "_base_executable", None)
    if type(executable) is not str or not Path(executable).is_file():
        raise ValueError("native transfer requires existing physical base interpreter")
    repository = Path(__file__).resolve().parents[2]
    module = (
        "src.adapters.windows_transfer_worker"
        if role == "idle"
        else "src.adapters.native_transfer_worker"
    )
    command = [executable, "-u", "-B", "-X", "utf8", "-m", module]
    if role != "idle":
        command.extend(("--role", role))
    if pause_after_train:
        _require(role == "new", "native pause requires replacement role")
        command.append("--pause-after-train")
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    env["PYTHONPATH"] = os.pathsep.join((str(repository), sysconfig.get_path("purelib")))
    peer = stack.enter_context(
        BoundedJsonChild(
            command,
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
        "native transfer hello differs from owned physical process",
    )
    return peer


def _report_only(before, after):
    validate_authority_floor(before, after)
    _require(
        after
        == replace(
            before,
            metadata=replace(
                before.metadata,
                sequence=after.metadata.sequence,
                observed_ns=after.metadata.observed_ns,
                usage=replace(
                    before.metadata.usage, peak_rss_bytes=after.metadata.usage.peak_rss_bytes
                ),
            ),
        ),
        "native transfer response changed non-report authority",
    )


def _budget_exact(value, *, models=0, updates=0, predictions=0):
    expected = dict(
        constructions=models,
        train_calls=updates,
        native_sgd=updates,
        raw_scaling_sgd=0,
        energy=4 * updates,
        predictions=predictions,
        split=0,
        guarded_inference=0,
        popen=0,
    )
    _require(
        type(value) is dict
        and value == expected
        and all(type(item) is int for item in value.values()),
        "native child counters differ from frozen admitted work",
    )


def _adopt(peer, path, known):
    peer.send(
        dict(
            op="adopt",
            authority_utf8=encode_authority(known).decode(),
            journal_path=str(path),
            binding=asdict(BINDING),
            limits=asdict(LIMITS),
        )
    )
    response = peer.receive()
    _require(
        set(response) == {"event", "authority_utf8", "budget"} and response["event"] == "adopted",
        "native adoption response differs",
    )
    adopted = decode_authority(response["authority_utf8"].encode())
    _report_only(known, adopted)
    _budget_exact(response["budget"])
    return adopted


def _update(peer, source, target, before, fixture, input_name, output_name):
    peer.send(
        dict(
            op="update",
            source=str(source),
            target=str(target),
            expected_source_sha256=fixture["digests"][input_name],
            expected_target_sha256=fixture["digests"][output_name],
        )
    )
    response = peer.receive()
    _require(
        set(response)
        == {
            "event",
            "authority_utf8",
            "completion",
            "before_payload_utf8",
            "payload_utf8",
            "digest",
            "energy",
            "predictions",
            "budget",
        }
        and response["event"] == "updated",
        "native update response schema differs",
    )
    proof = completion_from_wire(response["completion"])
    _budget_exact(
        response["budget"], models=1, updates=1, predictions=1 if input_name == "a" else 2
    )
    _require(
        proof.admission.expected == before,
        "native admission did not bind independently retained pre-update authority",
    )
    actual = decode_authority(response["authority_utf8"].encode())
    _report_only(proof.proposed, actual)
    _require(
        proof.proposed.metadata.manifest.native_sha256 == fixture["digests"][output_name],
        "native completion checkpoint binding differs",
    )
    _require(
        response["before_payload_utf8"].encode() == fixture["payloads"][input_name]
        and response["payload_utf8"].encode() == fixture["payloads"][output_name],
        "native complete snapshot differs from uninterrupted reference",
    )
    _require(
        response["digest"] == fixture["digests"][output_name]
        and response["energy"] == fixture["energies"][output_name]
        and response["predictions"] == fixture["predictions"][output_name],
        "native update diagnostics differ from fixed reference",
    )
    codec = CoupledCheckpointCodec()
    loaded = BoundedCheckpointFileStore(target, codec).load(
        binding=BINDING, expected_sha256=fixture["digests"][output_name], limits=LIMITS
    )
    _require(
        codec.encode(loaded, binding=BINDING, limits=LIMITS) == fixture["payloads"][output_name],
        "parent independently loaded native file differs",
    )
    return actual, response


def _restore(peer, path, before, fixture):
    peer.send(dict(op="restore", source=str(path), expected_sha256=fixture["digests"]["b"]))
    response = peer.receive()
    _require(
        set(response)
        == {"event", "authority_utf8", "payload_utf8", "digest", "predictions", "budget"}
        and response["event"] == "restored",
        "native restore response differs",
    )
    actual = decode_authority(response["authority_utf8"].encode())
    _budget_exact(response["budget"], models=1, predictions=1)
    validate_authority_floor(before, actual)
    _require(
        actual
        == replace(
            before,
            metadata=replace(
                before.metadata,
                sequence=actual.metadata.sequence,
                observed_ns=actual.metadata.observed_ns,
                usage=replace(
                    before.metadata.usage,
                    copied_bytes=before.metadata.usage.copied_bytes + 2 * LIMITS.max_encoded_bytes,
                    checkpoint_attempts=before.metadata.usage.checkpoint_attempts + 1,
                    peak_rss_bytes=actual.metadata.usage.peak_rss_bytes,
                ),
            ),
        ),
        "restore changed original bindings or wrong cumulative copy/attempt charge",
    )
    _require(
        response["payload_utf8"].encode() == fixture["payloads"]["b"]
        and response["digest"] == fixture["digests"]["b"]
        and response["predictions"] == fixture["predictions"]["b"],
        "restored completeB/predictions differ",
    )
    return actual, response


def _known(observation, anchor, worker, digest, family):
    absent = sha256(b"absent managed components in raw native transfer fixture").hexdigest()
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
    return AuthorityRecord(
        RecoveryMetadata(
            1,
            f"native-transfer-{family}-{anchor.pid}",
            observation.clock_epoch,
            "native-original",
            0,
            0,
            observation.now_ns,
            observation.now_ns,
            0,
            manifest,
            RecoveryLimits(2, 60_000_000_000, 10 * LIMITS.max_encoded_bytes, 0, 4, 512 * 1024**2),
            RecoveryUsage(0, 0, 0, 0, 0, observation.peak_rss_bytes),
            False,
            False,
        ),
        anchor,
        worker,
    )


def _actor_unchanged(fixture):
    _require(
        CoupledCheckpointCodec().encode(
            fixture["actor"].snapshot_state(), binding=BINDING, limits=LIMITS
        )
        == fixture["payloads"]["a"],
        "native candidate recovery changed actor fullA",
    )
    _require(
        fixture["actor"].predict_proba(fixture["x"].copy()).tolist() == fixture["actor_prediction"],
        "native candidate recovery changed actor prediction",
    )


def _complete_replacement(peer, paths, restored_b, fixture, path):
    completed_c, new_result = _update(peer, paths["b"], paths["c"], restored_b, fixture, "b", "c")
    peer.send(dict(op="exhaust"))
    exhausted = peer.receive()
    _require(
        set(exhausted) == {"event", "authority_utf8", "reason", "callbacks", "budget"}
        and exhausted["event"] == "exhausted"
        and exhausted["reason"] == "original authority allowance exceeded"
        and type(exhausted["callbacks"]) is int
        and exhausted["callbacks"] == 0,
        "native original quota was not refused before callbacks",
    )
    terminal = decode_authority(exhausted["authority_utf8"].encode())
    _budget_exact(exhausted["budget"], models=1, updates=1, predictions=2)
    validate_authority_floor(completed_c, terminal)
    _require(
        terminal
        == replace(
            completed_c,
            metadata=replace(
                completed_c.metadata,
                sequence=terminal.metadata.sequence,
                observed_ns=terminal.metadata.observed_ns,
                stopped=True,
                usage=replace(
                    completed_c.metadata.usage,
                    peak_rss_bytes=terminal.metadata.usage.peak_rss_bytes,
                ),
            ),
        ),
        "terminal native quota refusal changed protected state",
    )
    _require(
        terminal.metadata.usage.updates_admitted == terminal.metadata.usage.updates_completed == 2
        and terminal.metadata.usage.copied_bytes == 10 * LIMITS.max_encoded_bytes
        and terminal.metadata.usage.checkpoint_attempts == 4,
        "original cumulative native allowance differs",
    )
    _require(
        SqliteRecoveryJournal(path, terminal).read() == terminal,
        "native terminal checkpoint authority was not persisted",
    )
    return (
        dict(
            completed_c=asdict(completed_c),
            terminal=asdict(terminal),
            new_update=new_result,
            exhausted=exhausted,
        ),
        terminal,
    )


def _case(directory, family, *, interrupt_after_train=False):
    # Keep the single ownership transaction visible across the process crash;
    # computation, file validation and each channel operation live in helpers.
    directory.mkdir()
    fixture = _fixtures(family)
    paths = {name: directory / f"{name}.json" for name in ("a", "b", "c")}
    codec = CoupledCheckpointCodec()
    _require(
        BoundedCheckpointFileStore(paths["a"], codec).save(
            fixture["states"]["a"], binding=BINDING, limits=LIMITS
        )
        == fixture["digests"]["a"],
        "A fixture file binding differs",
    )
    path = directory / "authority.sqlite"
    with ExitStack() as stack:
        api = WindowsRecoveryApi()
        own = stack.enter_context(WindowsProcessHandle.pin(api.current_pid(), api=api))
        old_peer, old_worker_peer = _child(stack, "old"), _child(stack, "idle")
        old_anchor = stack.enter_context(WindowsProcessHandle.pin(old_peer.pid, api=api))
        old_worker = stack.enter_context(WindowsProcessHandle.pin(old_worker_peer.pid, api=api))
        observation = _initial_observation(old_peer, old_worker, api)
        _require(
            observation.observer == old_anchor.identity,
            "original native observer differs from owned coordinator",
        )
        initial = _known(
            observation, old_anchor.identity, old_worker.identity, fixture["digests"]["a"], family
        )
        SqliteRecoveryJournal.create(path, initial)
        adopted_a = _adopt(old_peer, path, initial)
        completed_b, old_result = _update(
            old_peer, paths["a"], paths["b"], adopted_a, fixture, "a", "b"
        )
        _actor_unchanged(fixture)
        # This floor comes from validated completion plus independent reference B,
        # never the manifest asserted by an arbitrary newer disk record.
        journal = SqliteRecoveryJournal(path, completed_b)
        _require(journal.read() == completed_b, "completedB witness and persisted authority differ")
        supervisor = stack.enter_context(
            WindowsTransferSupervisor(completed_b, own, old_anchor, old_worker)
        )
        _crash_original(old_peer)
        old_worker_peer.send(dict(op="exit"))
        _require(
            old_worker_peer.wait() == 0 and old_worker_peer.stderr == b"",
            "old native worker failed to exit",
        )
        _require(
            old_anchor.is_ended() and old_worker.is_ended(),
            "original native pair did not physically end",
        )
        _actor_unchanged(fixture)
        new_peer = _child(stack, "new", pause_after_train=interrupt_after_train)
        new_worker_peer = _child(stack, "idle")
        new_anchor = stack.enter_context(WindowsProcessHandle.pin(new_peer.pid, api=api))
        new_worker = stack.enter_context(WindowsProcessHandle.pin(new_worker_peer.pid, api=api))
        change = supervisor.plan_transfer(new_peer, new_anchor, new_worker, "native-replacement")
        try:
            _require(
                supervisor.commit_transfer(journal) == change,
                "native transfer CAS not acknowledged",
            )
            adopted_b = _adopt(new_peer, path, change.proposed)
            restored_b, restore_result = _restore(new_peer, paths["b"], adopted_b, fixture)
            outcome, terminal = (
                interrupt_replacement(
                    new_peer,
                    paths,
                    restored_b,
                    fixture,
                    path,
                    own,
                    new_anchor,
                    new_worker,
                    supervisor,
                )
                if interrupt_after_train
                else _complete_replacement(new_peer, paths, restored_b, fixture, path)
            )
            _require(
                terminal.metadata.observed_ns - initial.metadata.started_ns
                < initial.metadata.limits.max_elapsed_ns
                and terminal.metadata.usage.peak_rss_bytes <= initial.metadata.limits.max_rss_bytes,
                "native physical control exceeded original time/RSS",
            )
        except BaseException as error:
            try:
                SqliteRecoveryJournal.reconcile_coordinator_transfer(
                    path, supervisor.failure_witness()
                )
            except BaseException as reconciliation:
                error.add_note(
                    f"terminal native transfer reconciliation refused/failed: {reconciliation!r}"
                )
            raise
        _actor_unchanged(fixture)
        peers_to_exit = (new_worker_peer,) if interrupt_after_train else (new_peer, new_worker_peer)
        for peer in peers_to_exit:
            peer.send(dict(op="exit"))
            _require(
                peer.wait() == 0 and peer.stderr == b"", "replacement native pair failed to exit"
            )
        _require(
            new_anchor.is_ended() and new_worker.is_ended(),
            "replacement native processes did not end",
        )
        row = dict(
            family=family,
            initial=asdict(initial),
            adopted_a=asdict(adopted_a),
            completed_b=asdict(completed_b),
            transfer=asdict(change),
            adopted_b=asdict(adopted_b),
            restored_b=asdict(restored_b),
            old_update=old_result,
            restore=restore_result,
            **outcome,
            reference_payloads={name: raw.decode() for name, raw in fixture["payloads"].items()},
            actor_unchanged=True,
            returncodes=[
                p.returncode for p in (old_peer, old_worker_peer, new_peer, new_worker_peer)
            ],
        )
        _require(
            len(json.dumps(row, separators=(",", ":"), allow_nan=False).encode()) <= 65536,
            "native family summary exceeds frozen output cap",
        )
        return row


def run_example(directory: Path, *, interrupt_after_train: bool = False) -> dict[str, Any]:
    if sys.platform != "win32":
        raise OSError("native physical transfer requires Windows")
    _require(type(interrupt_after_train) is bool, "native interruption mode must be an exact flag")
    directory = Path(directory).resolve()
    _require(directory.is_dir(), "native transfer requires existing private directory")
    rows = [
        _case(directory / family, family, interrupt_after_train=interrupt_after_train)
        for family in FAMILIES
    ]
    parameter_fields = ("hidden_weights", "hidden_biases", "output_weight", "output_bias")
    for name in ("a", "b", "c"):
        pc = json.loads(rows[0]["reference_payloads"][name])["arrays"]
        neutral = json.loads(rows[1]["reference_payloads"][name])["arrays"]
        _require(
            all(pc[field] == neutral[field] for field in parameter_fields),
            "PC/neutral raw parameter frames differ",
        )
    return dict(
        protocol=describe_example(interrupt_after_train=interrupt_after_train),
        cases=rows,
        pc_neutral_parameters_exact=True,
        native_checkpoint_continuation_verified=not interrupt_after_train,
        native_replacement_interruption_verified=interrupt_after_train,
        full_managed_recovery_verified=False,
    )
