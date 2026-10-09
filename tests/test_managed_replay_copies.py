"""Actual checkpoint controller paths with fake learners and modeled byte ports."""

from copy import deepcopy
from hashlib import sha256
import gc
import pickle
from weakref import ref
from typing import Any
import pytest

from src.app.candidate_checkpoint import CandidateCheckpointController
from src.app.managed_data_lifecycle import ManagedDataLifecycle
from src.app.managed_replay_copies import ManagedReplayCopies
from src.app.managed_replay_origins import ManagedReplayOrigins
from src.core.data_erasure import ReplayPayloadErasure
from src.core.data_retention import DataRetentionPolicy
from src.core.native_model_copy import begin_model_copy
from src.core.payload_bytes import PayloadCopyLimits
from src.core.payload_ownership import PayloadOwnershipLimits
from src.core.replay_origin import ReplayOriginLimits
from test_managed_replay_origins import (
    setup as original_setup,
    queue,
    PORTS,
    LIMITS,
    WINDOW,
    Learner,
    Snapshot,
    Payload,
)


class Builder:
    def __init__(self, source):
        self.source = source
        self.calls = 0

    def __call__(self, state):
        self.calls += 1
        return self.source.fork()


def builder_source(builder):
    if type(builder) is not Builder:
        raise ValueError("unsupported fake original builder")
    return builder.source


def state_digest(state):
    return sha256(pickle.dumps(state)).hexdigest()


def footprint(learner):
    rows = len(learner.model.rows)
    return ReplayPayloadErasure(rows, rows, 32 * rows)


def setup(monkeypatch, *, limits=LIMITS, raw_limit=4096):
    owner, unused, runtime, clock, gate, budget = original_setup()
    del unused
    seconds = [0.0]
    budget.clock = lambda: seconds[0]
    copied = []
    snapshots = []
    bindings: dict[str, Any] = {}

    def fork(learner):
        window = begin_model_copy(learner.model)
        if window is None:
            return Learner(deepcopy(learner.model))
        try:
            life = bindings["life"]
            assert all(
                lock.locked()
                for lock in [
                    owner._gate,
                    runtime._write_gate,
                    life._registry._gate,
                    life._time_gate,
                    life._copy_budget._gate,
                    bindings["ledger"]._gate,
                    bindings["controller"]._gate,
                ]
            )
            copied.append(life._copy_budget._charged)
            memo: dict[int, Any] = {}
            model = deepcopy(learner.model, memo)
            window.copied(model, memo)
            return Learner(model)
        finally:
            window.end()

    def snapshot(learner):
        snapshots.append(learner)
        return deepcopy(learner.model)

    monkeypatch.setattr(Learner, "fork", fork)
    monkeypatch.setattr(Learner, "snapshot_state", snapshot)
    life = ManagedDataLifecycle(
        owner,
        policy=DataRetentionPolicy(
            4096,
            120,
            PayloadOwnershipLimits(8, 12),
            owned_payload_copies=PayloadCopyLimits(raw_limit),
            max_retention_seconds=1000.0,
        ),
        measure_payload_bytes=lambda _: 32,
        native_footprint=footprint,
        native_erase=lambda _: pytest.fail("erasure outside fake scope"),
        measure_auxiliary_bytes=lambda _: 0,
        measure_checkpoint_bytes=lambda view: 32 * len(view.state.rows),
        native_growth_bytes=lambda *_: 64,
        prepare_model_bytes=lambda builder, state: (
            footprint(builder.source).payload_bytes + 32 * len(state.rows)
        ),
        prediction_cache_bytes=lambda *_: 32,
    )
    ledger = ManagedReplayOrigins(owner, PORTS, limits, WINDOW)
    queue(owner, clock)
    ledger.train_ready()
    gate.pause()
    builder = Builder(runtime._candidate)
    # A real preparation failure after _models.append;no fake holder insertion.
    controller = CandidateCheckpointController(
        owner._shared,
        build_learner=builder,
        state_digest=state_digest,
        policy_digest=lambda learner: "0" * 64 if learner is runtime._candidate else "1" * 64,
    )
    token = controller.capture()
    copies = ManagedReplayCopies(ledger, builder_source=builder_source)
    bindings.update(life=life, ledger=ledger, controller=controller)
    return (
        owner,
        ledger,
        runtime,
        clock,
        life,
        seconds,
        controller,
        token,
        copies,
        copied,
        snapshots,
    )


def prepare(values):
    owner, ledger, runtime, clock, life, seconds, controller, token, copies, copied, snapshots = (
        values
    )
    with pytest.raises(ValueError, match="prepared native policy"):
        copies.restore(controller, token)
    assert len(controller._models) == 1 and controller._attempts == 1
    assert len(copied) == 1 and len(snapshots) == 2
    return controller._models[0]


def test_should_bind_actual_retained_failed_preparation_and_charge_before_copy(monkeypatch):
    values = setup(monkeypatch)
    owner, ledger, runtime, _, life, _, controller, token, copies, copied, _ = values
    before = ledger.accounting()
    raw = life._copy_budget._charged
    learner = prepare(values)
    after = ledger.accounting()
    assert after.records_created == before.records_created + 1
    assert after.invocations_started == before.invocations_started + 1
    assert after.live_records == before.live_records + 1
    assert copied == [raw + 128 + 32] and life._copy_budget._charged == copied[0]
    assert copies.origins(controller, learner)[0].key == ("e1", "s1")
    assert runtime._budget.updates_completed == 1 and ledger.origins()[0].update_number == 1
    assert controller._pending[token].owner is runtime
    assert learner.model.rows[0] is not runtime._candidate.model.rows[0]
    assert learner.model.rows[0].features is not runtime._candidate.model.rows[0].features


@pytest.mark.parametrize(
    "failure",
    [
        "revoked",
        "optout",
        "age",
        "elapsed",
        "rewind",
        "guard",
        "tombstone",
        "receipt",
        "source_payload",
        "untracked",
        "builder",
        "unenrolled",
        "copied_token",
    ],
)
def test_should_refuse_original_source_or_holder_changes_before_constructor_copy(
    monkeypatch, failure
):
    values = setup(monkeypatch)
    owner, ledger, runtime, clock, life, seconds, controller, token, copies, copied, _ = values
    before = ledger.accounting()
    if failure == "revoked":
        owner._revoked_keys.add(("e1", "s1"))
    elif failure == "optout":
        owner._opted_out.add("person-1")
    elif failure == "age":
        clock.advance_to(120)
    elif failure == "elapsed":
        seconds[0] = 1000.0
    elif failure == "rewind":
        seconds[0] = -1.0
    elif failure == "guard":
        runtime._inbox._training_guard = lambda *_: True
    elif failure == "tombstone":
        runtime._inbox._erased[("e1", "s1")] = object()
    elif failure == "receipt":
        runtime._inbox._applied[("e1", "s1")] = deepcopy(runtime._inbox._applied[("e1", "s1")])
    elif failure == "source_payload":
        runtime._candidate.model.rows[0].features.value = "changed"
    elif failure == "untracked":
        runtime._candidate.model.rows.append(Snapshot(Payload("x"), Payload("0")))
    elif failure == "builder":
        controller._build = Builder(Learner())
    elif failure == "unenrolled":
        life._registry._holders.pop(3)
    else:
        token = deepcopy(token)
    with pytest.raises(ValueError):
        copies.restore(controller, token)
    assert not copied and not controller._models and ledger.accounting() == before
    assert runtime._budget.updates_completed == 1


@pytest.mark.parametrize(
    "failure",
    [
        "target_payload",
        "target_row",
        "target_model",
        "model_list",
        "builder",
        "attempt",
        "reservation",
        "port",
        "revocation",
        "unenrolled",
        "policy",
        "probe",
        "copy_budget",
        "copy_limits",
        "raw_rewind",
    ],
)
def test_should_refuse_changed_copied_rows_and_original_retention_history(monkeypatch, failure):
    values = setup(monkeypatch)
    owner, ledger, runtime, _, life, _, controller, _, copies, _, _ = values
    learner = prepare(values)
    assert copies.origins(controller, learner)[0].subject_id == "person-1"
    if failure == "target_payload":
        learner.model.rows[0].features.value = "changed"
    elif failure == "target_row":
        learner.model.rows[0] = deepcopy(learner.model.rows[0])
    elif failure == "target_model":
        learner.model = deepcopy(learner.model)
    elif failure == "model_list":
        controller._models = list(controller._models)
    elif failure == "builder":
        controller._build = Builder(runtime._candidate)
    elif failure == "attempt":
        controller._attempts = 0
    elif failure == "reservation":
        ledger._copy_slots = 0
    elif failure == "port":
        copies._builder_source = lambda _: runtime._candidate
    elif failure == "revocation":
        owner._revoked_keys.add(("e1", "s1"))
    elif failure == "policy":
        controller._policy = lambda _: "0" * 64
    elif failure == "probe":
        controller._digest = lambda _: "0" * 64
    elif failure == "copy_budget":
        from src.app.payload_copy_budget import PayloadCopyBudget

        life._copy_budget = PayloadCopyBudget(life._policy.owned_payload_copies)
    elif failure == "copy_limits":
        life._copy_budget._limits = PayloadCopyLimits(4096)
    elif failure == "raw_rewind":
        life._copy_budget._charged -= 1
    else:
        life._registry._holders.pop(3)
    with pytest.raises(ValueError):
        copies.origins(controller, learner)
    assert runtime._budget.updates_completed == 1


@pytest.mark.parametrize("cap", ["live", "created", "invocations", "metadata", "raw"])
def test_should_keep_failed_admission_charges_and_refuse_allocation(monkeypatch, cap):
    config = dict(vars(LIMITS))
    raw_limit = 4096
    if cap == "live":
        config["max_live_records"] = 1
    elif cap == "created":
        config["max_records_created"] = 1
    elif cap == "invocations":
        config["max_invocations"] = 1
    elif cap == "metadata":
        config["max_metadata_bytes"] = 4096
    else:
        raw_limit = 383
    values = setup(monkeypatch, limits=ReplayOriginLimits(**config), raw_limit=raw_limit)
    _, ledger, _, _, life, _, controller, token, copies, copied, _ = values
    before = ledger.accounting()
    raw = life._copy_budget._charged
    with pytest.raises(ValueError):
        copies.restore(controller, token)
    assert not copied and not controller._models and controller._attempts == 1
    after = ledger.accounting()
    assert after.invocations_started >= before.invocations_started
    assert after.metadata_bytes_charged >= before.metadata_bytes_charged
    assert life._copy_budget._charged == raw + 128
    assert not life._time_gate.locked() and not life._copy_budget._gate.locked()


def test_should_release_weak_target_without_refunding_original_ledger_reservations(monkeypatch):
    values = setup(monkeypatch)
    _, ledger, _, _, life, _, controller, _, copies, _, _ = values
    learner = prepare(values)
    weak_model = ref(learner.model)
    weak_row = ref(learner.model.rows[0])
    before = ledger.accounting()
    raw = life._copy_budget._charged
    controller._models.clear()
    del learner
    gc.collect()
    assert weak_model() is None and weak_row() is None
    assert ledger.accounting() == before and life._copy_budget._charged == raw


def test_should_not_renew_shared_metadata_by_constructing_another_coordinator(monkeypatch):
    values = setup(monkeypatch)
    _, ledger, _, _, _, _, controller, token, copies, _, _ = values
    learner = prepare(values)
    other = ManagedReplayCopies(ledger, builder_source=builder_source)
    with pytest.raises(ValueError, match="original retained"):
        other.origins(controller, learner)
    before = ledger.accounting()
    with pytest.raises(ValueError, match="prepared native policy"):
        other.restore(controller, token)
    assert ledger.accounting().records_created == before.records_created + 1
    assert ledger.accounting().live_records == before.live_records + 1
    assert ledger._copy_slots == 2


def test_should_refuse_consent_changed_by_footprint_observation_before_copy(monkeypatch):
    values = setup(monkeypatch)
    owner, ledger, _, _, life, _, controller, token, copies, copied, _ = values
    original = life._footprint

    def changed(learner):
        result = original(learner)
        if life._time_gate.locked():
            owner._revoked_keys.add(("e1", "s1"))
        return result

    life._footprint = changed
    before = ledger.accounting()
    with pytest.raises(ValueError, match="revoked"):
        copies.restore(controller, token)
    assert not copied and not controller._models
    assert ledger.accounting().metadata_bytes_charged > before.metadata_bytes_charged
    assert not life._time_gate.locked() and not life._copy_budget._gate.locked()
