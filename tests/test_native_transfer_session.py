"""Charged file/SQLite ordering with trusted fake models; no native execution."""

from contextlib import suppress
from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import src.adapters.native_transfer_session as session_module
from src.adapters.native_transfer_protocol import (
    BINDING,
    LIMITS,
    completion_from_wire,
    completion_to_wire,
)
from src.adapters.native_transfer_session import CompletionJournal, NativeTransferSession
from src.adapters.coupled_checkpoint_codec import CoupledCheckpointCodec
from src.core.checkpoint_codec import CodecBinding
from src.core.coordinator_transfer import (
    CoordinatorTransferObservation,
    SupervisorRecoveryWitness,
    plan_coordinator_transfer,
)
from src.core.coupled_circadian_predictive_coding import CoupledCircadianConfig, _plasticity_gates
from src.core.coupled_energy import CoupledParameters
from src.core.coupled_predictive_coding import CoupledSnapshot, CoupledTrainResult, CoupledWakeState
from src.core.recovery_admission import RecoveryLimits, RecoveryUsage
from src.core.recovery_authority import plan_reservation, validate_authority_floor
from src.core.recovery_completion import plan_completion, validate_authority_completion
from src.core.recovery_observation import (
    RecoveryHostObservation,
    RecoveryProcessIdentity,
    anchored_clock_epoch,
)
from src.core.recovery_reporting import plan_report
from src.infra.checkpoint_files import BoundedCheckpointFileStore
from src.infra.recovery_authority_codec import decode_authority, encode_authority
from src.infra.sqlite_recovery_journal import SqliteRecoveryJournal
from test_recovery_authority import fixture as authority_fixture
from test_recovery_coordinator import Observer, Probe


def snapshot(family="pc"):
    parameters = CoupledParameters(
        (
            np.arange(6, dtype=np.float64).reshape(2, 3) / 20,
            np.arange(6, dtype=np.float64).reshape(3, 2) / 30,
        ),
        (np.array([[0.1, -0.2, 0.3]]), np.array([[0.2, -0.1]])),
        np.array([[0.4], [-0.2]]),
        np.array([[-0.0]]),
    )
    wake = None
    if family != "pc":
        config = CoupledCircadianConfig(min_plasticity=1.0 if family == "neutral_cpc" else 0.2)
        chemical = np.array([0.1, 0.2])
        wake = CoupledWakeState(
            config.chemical_decay,
            config.chemical_buildup_rate,
            config.plasticity_sensitivity,
            config.min_plasticity,
            chemical,
            _plasticity_gates(chemical, config),
            2,
        )
    return CoupledSnapshot(1, parameters, (1.0, 1.0), wake)


def advance(value):
    result = deepcopy(value)
    result.parameters.output_bias[0, 0] += 0.1
    result.parameters.hidden_weights[0][0, 0] += 0.01
    wake = result.wake_state
    if wake is not None:
        config = CoupledCircadianConfig(
            wake.chemical_decay,
            wake.chemical_buildup_rate,
            wake.plasticity_sensitivity,
            wake.min_plasticity,
        )
        chemical = wake.chemical_state + 0.05
        result = replace(
            result,
            wake_state=replace(
                wake,
                chemical_state=chemical,
                plasticity_state=_plasticity_gates(chemical, config),
                wake_updates=wake.wake_updates + 1,
            ),
        )
    return result


def encode(value):
    return CoupledCheckpointCodec().encode(value, binding=BINDING, limits=LIMITS)


def digest(value):
    return sha256(encode(value)).hexdigest()


def known_for(value):
    record, _ = authority_fixture()
    manifest = replace(
        record.metadata.manifest,
        source_sha256=BINDING.source_sha256,
        policy_sha256=BINDING.policy_sha256,
        payload_sha256=digest(value),
        native_sha256=digest(value),
    )
    return replace(
        record,
        metadata=replace(
            record.metadata,
            manifest=manifest,
            limits=RecoveryLimits(2, 100000, 10 * LIMITS.max_encoded_bytes, 0, 4, 2048),
            usage=RecoveryUsage(0, 0, 0, 0, 0, 512),
        ),
    )


def proof_fixture():
    known = known_for(snapshot())
    observation = RecoveryHostObservation(
        known.metadata.clock_epoch, 600, 640, 640, known.anchor, known.worker, False
    )
    admission = plan_reservation(
        known,
        observation,
        updates=1,
        copy_bytes=4 * LIMITS.max_encoded_bytes,
        checkpoint_attempts=1,
    )
    pending = plan_report(admission.proposed, replace(observation, now_ns=610)).proposed
    manifest = replace(known.metadata.manifest, native_sha256="b" * 64, payload_sha256="b" * 64)
    return plan_completion(admission, pending, replace(observation, now_ns=620), manifest)


class FakeModel:
    def __init__(self, value, h):
        self.value, self.h = deepcopy(value), h

    def snapshot_state(self):
        self.h.check("snapshot")
        return deepcopy(self.value)

    def train_epoch(
        self, features, targets, learning_rate, inference_steps, inference_learning_rate
    ):
        self.h.check("train")
        assert np.array_equal(features, [[0.2, -0.3], [-0.4, 0.6]])
        assert np.array_equal(targets, [[1.0], [0.0]])
        assert (learning_rate, inference_steps, inference_learning_rate) == (0.02, 2, 0.05)
        self.value = advance(self.value)
        features.fill(99)
        targets.fill(99)
        return CoupledTrainResult(0.75)

    def predict_proba(self, features):
        self.h.check("predict")
        assert np.array_equal(features, [[0.2, -0.3], [-0.4, 0.6]])
        return np.full((2, 1), 0.5 + self.value.parameters.output_bias[0, 0] / 10)


@pytest.fixture
def harness(tmp_path, monkeypatch):
    value = snapshot()
    h = SimpleNamespace(
        root=tmp_path,
        value=value,
        known=known_for(value),
        events=[],
        models=[],
        sessions=[],
        session=None,
        mode="update",
        completed=0,
        copied=0,
        attempts=0,
    )
    h.source, h.target, h.third, h.path = (
        tmp_path / name for name in ("a.bin", "b.bin", "c.bin", "authority.sqlite")
    )
    BoundedCheckpointFileStore(h.source, CoupledCheckpointCodec()).save(
        value, binding=BINDING, limits=LIMITS
    )
    SqliteRecoveryJournal.create(h.path, h.known)

    def check(stage):
        current = h.session.authority()
        usage = current.metadata.usage
        assert usage.updates_completed == h.completed
        assert usage.updates_admitted == h.completed + (h.mode == "update")
        assert current.metadata.uncertain_work is (h.mode == "update")
        assert (
            usage.copied_bytes
            == h.copied + (4 if h.mode == "update" else 2) * LIMITS.max_encoded_bytes
        )
        assert usage.checkpoint_attempts == h.attempts + 1
        h.events.append(stage)

    h.check = check

    def factory(saved):
        h.check("factory")
        model = FakeModel(saved, h)
        h.models.append(model)
        return model

    h.factory = factory

    def adopt(record):
        log: list[tuple[str, object]] = []
        observer = Observer(record, log)
        observer.now = max(600, record.metadata.observed_ns)
        observer.hook = lambda observation: replace(
            observation,
            rss_bytes=1024,
            peak_rss_bytes=max(1024, record.metadata.usage.peak_rss_bytes),
        )
        session = NativeTransferSession(
            record,
            h.path,
            observer,
            Probe(record.anchor, log),
            Probe(record.worker, log),
            model_factory=h.factory,
        )
        h.sessions.append(session)
        h.session = session
        return session

    h.adopt = adopt
    original_load, original_save = BoundedCheckpointFileStore.load, BoundedCheckpointFileStore.save

    def load(store, *args, **kwargs):
        h.check("load")
        return original_load(store, *args, **kwargs)

    def save(store, *args, **kwargs):
        h.check("save")
        return original_save(store, *args, **kwargs)

    monkeypatch.setattr(BoundedCheckpointFileStore, "load", load)
    monkeypatch.setattr(BoundedCheckpointFileStore, "save", save)
    try:
        yield h
    finally:
        for session in h.sessions:
            with suppress(ValueError):
                session.close()


def assert_report_only(actual, previous):
    assert (
        replace(
            actual,
            metadata=replace(
                actual.metadata,
                sequence=previous.metadata.sequence,
                observed_ns=previous.metadata.observed_ns,
                usage=replace(
                    actual.metadata.usage, peak_rss_bytes=previous.metadata.usage.peak_rss_bytes
                ),
            ),
        )
        == previous
    )


def test_should_roundtrip_exact_completion_wire_without_disk_bootstrap_or_authority_claim():
    proof = proof_fixture()
    wire = completion_to_wire(proof)
    assert wire == dict(
        admission_expected_utf8=encode_authority(proof.admission.expected).decode(),
        admission_proposed_utf8=encode_authority(proof.admission.proposed).decode(),
        admission_observation=asdict(proof.admission.observation),
        expected_utf8=encode_authority(proof.expected).decode(),
        proposed_utf8=encode_authority(proof.proposed).decode(),
        observation=asdict(proof.observation),
    )
    restored = completion_from_wire(deepcopy(wire))
    assert restored == proof
    validate_authority_completion(restored)
    with pytest.raises(ValueError):
        validate_authority_floor(proof.expected, proof.proposed)


@pytest.mark.parametrize(
    "fault",
    [
        "mapping",
        "extra",
        "missing",
        "bytes",
        "noncanonical",
        "bool_time",
        "bool_pid",
        "false_ended",
        "extra_identity",
        "missing_identity",
        "missing_host",
        "extra_host",
        "wrong_owner",
        "wrong_epoch",
        "wrong_manifest",
        "proposed_pending",
    ],
)
def test_should_refuse_malformed_or_unbound_completion_proof(fault):
    proof = proof_fixture()
    wire: Any = completion_to_wire(proof)
    if fault == "mapping":
        wire = SimpleNamespace(**wire)
    elif fault == "extra":
        wire["extra"] = 1
    elif fault == "missing":
        wire.pop("expected_utf8")
    elif fault == "bytes":
        wire["expected_utf8"] = wire["expected_utf8"].encode()
    elif fault == "noncanonical":
        wire["expected_utf8"] += " "
    elif fault == "bool_time":
        wire["observation"]["now_ns"] = True
    elif fault == "bool_pid":
        wire["observation"]["observer"]["pid"] = True
    elif fault == "false_ended":
        wire["observation"]["previous_owner_ended"] = 0
    elif fault == "extra_identity":
        wire["observation"]["observer"]["extra"] = 0
    elif fault == "missing_identity":
        wire["observation"]["observer"].pop("created_filetime")
    elif fault == "missing_host":
        wire["observation"].pop("peak_rss_bytes")
    elif fault == "extra_host":
        wire["observation"]["extra"] = 0
    elif fault == "wrong_owner":
        wire["observation"]["previous_owner"]["created_filetime"] = 999
    elif fault == "wrong_epoch":
        wire["observation"]["clock_epoch"] = "foreign"
    elif fault == "wrong_manifest":
        wire["proposed_utf8"] = encode_authority(
            replace(
                proof.proposed,
                metadata=replace(
                    proof.proposed.metadata,
                    manifest=replace(proof.proposed.metadata.manifest, actor_sha256="f" * 64),
                ),
            )
        ).decode()
    else:
        wire["proposed_utf8"] = wire["expected_utf8"]
    with pytest.raises(ValueError):
        completion_from_wire(wire)


@pytest.mark.parametrize("family", ["pc", "neutral_cpc", "active_cpc"])
def test_should_persist_admission_before_every_payload_or_model_operation_and_complete_owned_b(
    harness, family
):
    h = harness
    if family != "pc":
        h.value = snapshot(family)
        h.known = known_for(h.value)
        h.source.write_bytes(encode(h.value))
        # Replace this fixture's initial independent authority before adopting it.
        h.path.unlink()
        SqliteRecoveryJournal.create(h.path, h.known)
    session = h.adopt(h.known)
    adopted = session.authority()
    assert_report_only(adopted, h.known)
    assert h.events == []
    result = session.update(h.source, h.target, digest(h.value), digest(advance(h.value)))
    assert set(result) == {
        "event",
        "authority_utf8",
        "completion",
        "before_payload_utf8",
        "payload_utf8",
        "digest",
        "energy",
        "predictions",
    }
    assert result["event"] == "updated" and result["energy"] == 0.75
    assert result["before_payload_utf8"].encode() == encode(h.value)
    assert result["payload_utf8"].encode() == encode(advance(h.value)) == h.target.read_bytes()
    assert result["digest"] == digest(advance(h.value))
    assert result["predictions"] == [[0.51], [0.51]]
    completion = completion_from_wire(result["completion"])
    assert completion.admission.expected == adopted
    assert completion.admission.proposed.metadata.uncertain_work
    assert (
        completion.proposed.metadata.usage.updates_admitted
        == completion.proposed.metadata.usage.updates_completed
        == 1
    )
    final = decode_authority(result["authority_utf8"].encode())
    assert_report_only(final, completion.proposed)
    assert final == session.authority()
    assert (
        h.events.index("load")
        < h.events.index("factory")
        < h.events.index("train")
        < h.events.index("save")
    )
    assert (
        h.events.count("load") == 2
        and h.events.count("factory") == h.events.count("train") == h.events.count("save") == 1
    )
    h.models[0].value.parameters.output_bias.fill(99)
    assert h.target.read_bytes() == result["payload_utf8"].encode()


def test_should_charge_transferred_restore_then_continue_under_original_two_update_allowance(
    harness,
):
    h = harness
    old = h.adopt(h.known)
    result_b = old.update(h.source, h.target, digest(h.value), digest(advance(h.value)))
    b = decode_authority(result_b["authority_utf8"].encode())
    supervisor, anchor, worker = (RecoveryProcessIdentity(pid, pid * 10) for pid in (30, 40, 50))
    now = b.metadata.observed_ns
    host = lambda at, owner, ended: RecoveryHostObservation(
        anchored_clock_epoch(supervisor), at, 1024, 1024, supervisor, owner, ended
    )
    remote = RecoveryHostObservation(
        anchored_clock_epoch(anchor), now + 30, 1024, 1024, anchor, worker, False
    )
    change = plan_coordinator_transfer(
        SupervisorRecoveryWitness(b, supervisor),
        CoordinatorTransferObservation(
            host(now + 10, b.anchor, True),
            host(now + 20, b.worker, True),
            remote,
            host(now + 40, anchor, False),
        ),
        "replacement",
    )
    assert SqliteRecoveryJournal(h.path, b).transfer_coordinator(change)
    with pytest.raises(ValueError):
        old.close()
    h.completed, h.copied, h.attempts, h.mode = 1, 4 * LIMITS.max_encoded_bytes, 2, "restore"
    h.events.clear()
    new = h.adopt(change.proposed)
    adopted = new.authority()
    assert_report_only(adopted, change.proposed)
    restored = new.restore(h.target, digest(advance(h.value)))
    assert set(restored) == {"event", "authority_utf8", "payload_utf8", "digest", "predictions"}
    assert restored["event"] == "restored" and restored["payload_utf8"] == result_b["payload_utf8"]
    assert h.events.index("load") < h.events.index("factory") < h.events.index("predict")
    assert "train" not in h.events
    known_b = decode_authority(restored["authority_utf8"].encode())
    assert known_b.metadata.usage.copied_bytes == 6 * LIMITS.max_encoded_bytes
    assert known_b.metadata.usage.checkpoint_attempts == 3
    h.mode, h.copied, h.attempts = "update", 6 * LIMITS.max_encoded_bytes, 3
    h.events.clear()
    result_c = new.update(
        h.target, h.third, digest(advance(h.value)), digest(advance(advance(h.value)))
    )
    proof = completion_from_wire(result_c["completion"])
    assert proof.admission.expected == known_b
    assert h.events.count("factory") == 0 and h.events.count("train") == 1
    exhausted = new.refuse_update()
    assert exhausted["event"] == "exhausted" and exhausted["callbacks"] == 0
    terminal = decode_authority(exhausted["authority_utf8"].encode())
    assert terminal.metadata.stopped and not terminal.metadata.uncertain_work
    assert terminal.metadata.usage == RecoveryUsage(2, 2, 10 * LIMITS.max_encoded_bytes, 0, 4, 1024)
    assert terminal.metadata.limits == h.known.metadata.limits
    assert terminal.metadata.started_ns == h.known.metadata.started_ns
    assert terminal.metadata.manifest.actor_sha256 == h.known.metadata.manifest.actor_sha256


@pytest.mark.parametrize(
    "fault",
    [
        "binding_mapping",
        "binding_schema",
        "binding_hash",
        "binding_source",
        "binding_policy",
        "limits_mapping",
        "limits_schema",
        "limits_bool",
        "encoded_cap",
        "array_cap",
        "layer_cap",
        "dimension_cap",
        "managed",
        "uncertain",
        "stopped",
        "pending",
    ],
)
def test_should_refuse_constructor_controls_before_journal_observer_or_factory(
    tmp_path, monkeypatch, fault
):
    h = SimpleNamespace(known=known_for(snapshot()), path=tmp_path / "unused.sqlite")
    binding: Any = deepcopy(BINDING)
    limits: Any = deepcopy(LIMITS)
    known = h.known
    if fault == "binding_mapping":
        binding = asdict(binding)
    elif fault == "binding_schema":
        object.__setattr__(binding, "extra", 1)
    elif fault == "binding_hash":
        object.__setattr__(binding, "source_sha256", True)
    elif fault == "binding_source":
        binding = CodecBinding("f" * 64, binding.policy_sha256)
    elif fault == "binding_policy":
        binding = CodecBinding(binding.source_sha256, "f" * 64)
    elif fault == "limits_mapping":
        limits = asdict(limits)
    elif fault == "limits_schema":
        object.__delattr__(limits, "max_dimension")
    elif fault == "limits_bool":
        object.__setattr__(limits, "max_array_bytes", True)
    elif fault.endswith("_cap"):
        name = dict(
            encoded_cap="max_encoded_bytes",
            array_cap="max_array_bytes",
            layer_cap="max_layers",
            dimension_cap="max_dimension",
        )[fault]
        limits = replace(limits, **{name: getattr(limits, name) + 1})
    elif fault == "managed":
        known = replace(
            known,
            metadata=replace(
                known.metadata, manifest=replace(known.metadata.manifest, payload_sha256="e" * 64)
            ),
        )
    elif fault == "uncertain":
        known = replace(known, metadata=replace(known.metadata, uncertain_work=True))
    elif fault == "stopped":
        known = replace(known, metadata=replace(known.metadata, stopped=True))
    else:
        known = replace(
            known,
            metadata=replace(
                known.metadata,
                usage=replace(known.metadata.usage, updates_admitted=1),
                uncertain_work=True,
            ),
        )
    touched = []

    def forbidden(*args, **kwargs):
        touched.append("called")
        raise AssertionError("invalid constructor reached operational callback")

    monkeypatch.setattr(SqliteRecoveryJournal, "read", forbidden)
    observer = SimpleNamespace(observe=forbidden)
    probes = [Probe(known.anchor, []), Probe(known.worker, [])]
    with pytest.raises(ValueError):
        NativeTransferSession(
            known,
            h.path,
            observer,
            *probes,
            binding=binding,
            limits=limits,
            model_factory=forbidden,
        )
    assert touched == []
    assert not any(probe.closed for probe in probes)


def test_should_refuse_pure_update_preflight_without_spending_or_touching_payload(harness):
    from pathlib import Path

    h = harness
    session = h.adopt(h.known)
    before = session.authority()
    source, target = h.source, h.target
    source_hash, target_hash = digest(h.value), digest(advance(h.value))
    cases = [
        (Path("a.bin"), target, source_hash, target_hash),
        (source, Path("b.bin"), source_hash, target_hash),
        (str(source), target, source_hash, target_hash),
        (source, source, source_hash, target_hash),
        (source, target, True, target_hash),
        (source, target, "a", target_hash),
        (source, target, "A" * 64, target_hash),
        (source, target, "f" * 64, target_hash),
        (source, target, source_hash, False),
        (source, target, source_hash, "z" * 64),
    ]
    # Reusing the clean authority also proves a pure refusal does not poison it.
    for arguments in cases:
        with pytest.raises(ValueError):
            session.update(*arguments)
        assert session.authority() == before and h.events == []
        assert h.source.read_bytes() == encode(h.value) and not h.target.exists()


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "corrupt",
        "geometry",
        "precision",
        "factory",
        "forged_restore",
        "train",
        "candidate",
        "save",
        "validator",
        "output_digest",
        "predictions",
    ],
)
def test_should_keep_postadmission_faults_spent_uncertain_and_permanently_refuse_retry(
    harness, monkeypatch, fault
):
    h = harness
    if fault in ("geometry", "precision"):
        h.value = (
            replace(
                h.value,
                parameters=replace(
                    h.value.parameters,
                    hidden_weights=(np.zeros((3, 3)), h.value.parameters.hidden_weights[1]),
                ),
            )
            if fault == "geometry"
            else replace(h.value, precisions=(1.0, 2.0))
        )
        h.known = known_for(h.value)
        h.source.write_bytes(encode(h.value))
        h.path.unlink()
        SqliteRecoveryJournal.create(h.path, h.known)
    if fault == "factory":

        def factory(saved):
            h.check("factory")
            raise OSError("injected factory failure")

        h.factory = factory
    elif fault == "forged_restore":
        original_factory = h.factory

        def factory(saved):
            model = original_factory(saved)
            model.value.parameters.output_bias.fill(0.8)
            return model

        h.factory = factory
    session = h.adopt(h.known)
    if fault == "missing":
        h.source.unlink()
    elif fault == "corrupt":
        h.source.write_bytes(encode(h.value)[:-1] + b"!")
    elif fault == "train":
        original_train = FakeModel.train_epoch

        def train(model, *args, **kwargs):
            original_train(model, *args, **kwargs)
            raise OSError("injected after fake model mutation")

        monkeypatch.setattr(FakeModel, "train_epoch", train)
    elif fault == "candidate":
        original_train = FakeModel.train_epoch

        def train(model, *args, **kwargs):
            result = original_train(model, *args, **kwargs)
            model.value.parameters.output_bias.fill(np.nan)
            return result

        monkeypatch.setattr(FakeModel, "train_epoch", train)
    elif fault == "save":

        def save(*args, **kwargs):
            h.check("save")
            raise OSError("injected save failure")

        monkeypatch.setattr(BoundedCheckpointFileStore, "save", save)
    elif fault == "validator":
        original_load = BoundedCheckpointFileStore.load

        def load(store, *args, **kwargs):
            if h.events.count("load"):
                h.check("load")
                raise OSError("injected target validation failure")
            return original_load(store, *args, **kwargs)

        monkeypatch.setattr(BoundedCheckpointFileStore, "load", load)
    elif fault == "predictions":

        def predict(model, features):
            h.check("predict")
            return np.full((2, 1), 1.1)

        monkeypatch.setattr(FakeModel, "predict_proba", predict)
    expected = "f" * 64 if fault == "output_digest" else digest(advance(h.value))
    with pytest.raises((ValueError, OSError)):
        session.update(h.source, h.target, digest(h.value), expected)
    current = session.authority()
    assert current.metadata.stopped and current.metadata.uncertain_work
    assert current.metadata.manifest == h.known.metadata.manifest
    assert current.metadata.usage == RecoveryUsage(1, 0, 4 * LIMITS.max_encoded_bytes, 0, 1, 1024)
    events = list(h.events)
    with pytest.raises(ValueError):
        session.update(h.source, h.target, digest(h.value), expected)
    assert h.events == events and session.authority() == current


def test_should_record_only_acknowledged_actual_completion_and_reject_disk_asserted_floor(harness):
    h = harness
    proof = proof_fixture()
    assert proof.admission.expected == h.known
    journal = CompletionJournal(h.path, h.known)
    assert journal.last_completion is None
    assert journal.advance(proof.admission)
    assert journal.report(
        plan_report(proof.admission.proposed, replace(proof.admission.observation, now_ns=610))
    )
    assert journal.complete(proof)
    assert journal.last_completion == proof and journal.read() == proof.proposed
    with pytest.raises(ValueError):
        SqliteRecoveryJournal(h.path, h.known).read()


@pytest.mark.parametrize("outcome", ["false", "exception", "postcommit_exception"])
def test_should_not_manufacture_completion_proof_from_missing_acknowledgment(
    harness, monkeypatch, outcome
):
    h = harness
    proof = proof_fixture()
    journal = CompletionJournal(h.path, h.known)
    assert journal.advance(proof.admission)
    assert journal.report(
        plan_report(proof.admission.proposed, replace(proof.admission.observation, now_ns=610))
    )
    original = SqliteRecoveryJournal.complete

    def complete(current, change):
        if outcome == "false":
            return False
        if outcome == "postcommit_exception":
            assert original(current, change)
        raise OSError("unacknowledged completion")

    monkeypatch.setattr(SqliteRecoveryJournal, "complete", complete)
    if outcome == "false":
        assert journal.complete(proof) is False
    else:
        with pytest.raises(OSError):
            journal.complete(proof)
    assert journal.last_completion is None


def test_should_charge_failed_restore_and_never_reopen_it_or_retry_factory(harness, monkeypatch):
    h = harness
    default_factory = h.factory
    for fault in (
        "missing",
        "corrupt",
        "factory",
        "forged_restore",
        "predictions",
        "prediction_mutation",
    ):
        h.path.unlink()
        SqliteRecoveryJournal.create(h.path, h.known)
        h.source.write_bytes(encode(h.value))
        h.mode = "restore"
        h.events.clear()
        h.factory = default_factory
        with monkeypatch.context() as patch:
            if fault in ("factory", "forged_restore"):

                def factory(saved):
                    if fault == "factory":
                        h.check("factory")
                        raise OSError("restore factory failed")
                    model = default_factory(saved)
                    model.value.parameters.output_bias.fill(0.9)
                    return model

                h.factory = factory
            if fault in ("predictions", "prediction_mutation"):

                def predict(model, features):
                    h.check("predict")
                    if fault == "prediction_mutation":
                        model.value.parameters.output_bias.fill(0.9)
                        return np.full((2, 1), 0.5)
                    return np.full((1, 2), 0.5)

                patch.setattr(FakeModel, "predict_proba", predict)
            session = h.adopt(h.known)
            if fault == "missing":
                h.source.unlink()
            elif fault == "corrupt":
                h.source.write_bytes(b"damaged")
            with pytest.raises((ValueError, OSError)):
                session.restore(h.source, digest(h.value))
            current = session.authority()
            assert current.metadata.stopped and not current.metadata.uncertain_work
            assert current.metadata.usage == RecoveryUsage(
                0, 0, 2 * LIMITS.max_encoded_bytes, 0, 1, 1024
            )
            events = list(h.events)
            with pytest.raises(ValueError):
                session.restore(h.source, digest(h.value))
            assert h.events == events and session.authority() == current


def test_should_refuse_changed_retained_model_before_second_train_without_reloading_or_refund(
    harness,
):
    h = harness
    session = h.adopt(h.known)
    session.update(h.source, h.target, digest(h.value), digest(advance(h.value)))
    h.completed, h.copied, h.attempts = 1, 4 * LIMITS.max_encoded_bytes, 1
    h.models[0].value.parameters.output_bias.fill(0.9)
    h.events.clear()
    saved = h.target.read_bytes()
    with pytest.raises(ValueError):
        session.update(
            h.target, h.third, digest(advance(h.value)), digest(advance(advance(h.value)))
        )
    assert h.events == ["snapshot"]
    current = session.authority()
    assert current.metadata.stopped and current.metadata.uncertain_work
    assert current.metadata.usage == RecoveryUsage(2, 1, 8 * LIMITS.max_encoded_bytes, 0, 2, 1024)
    assert h.target.read_bytes() == saved and not h.third.exists()


def test_should_refuse_completed_result_without_retained_acknowledged_proof_and_not_repeat_work(
    harness, monkeypatch
):
    h = harness
    session = h.adopt(h.known)
    monkeypatch.setattr(
        session_module.CompletionJournal, "complete", SqliteRecoveryJournal.complete
    )
    with pytest.raises((ValueError, RuntimeError)):
        session.update(h.source, h.target, digest(h.value), digest(advance(h.value)))
    assert h.target.read_bytes() == encode(advance(h.value))
    current = session.authority()
    assert current.metadata.usage.updates_completed == 1
    events = list(h.events)
    with pytest.raises(ValueError):
        session.update(
            h.target, h.third, digest(advance(h.value)), digest(advance(advance(h.value)))
        )
    assert h.events == events
