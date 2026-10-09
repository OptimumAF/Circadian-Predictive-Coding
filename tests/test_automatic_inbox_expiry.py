"""Original direct expiry with genuine numeric fake work and birth cleanup ports.

Why this: automatic lineage publication must follow whole original cleanup,
while observation refusal cannot retain delivered raw data or renew allowances.
These tests perform no native/CPC work, GC or model snapshot/restore.
"""

from dataclasses import asdict, dataclass, replace
import json
from typing import Any, Callable
from weakref import ref

import pytest

import src.app.managed_data_lifecycle as lifecycle_module
import src.app.expiry_inbox_observation as observation_module
from src.app.actor_shadow import ActorShadowRuntime
from src.app.expiry_inbox_observation import ExpiryObservationError
from src.app.managed_data_lifecycle import ManagedDataLifecycle
from src.app.managed_experience import ManagedExperienceOwner
from src.app.managed_inbox_origins import ManagedInboxOrigins
from src.app.managed_lifecycle_capture import capture_managed_lifecycle
from src.app.managed_record_capture import capture_managed_records
from src.app.managed_replay_origins import ManagedReplayOrigins
from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
from src.app.serving_promotion import PromotableActor
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.data_lifecycle import LifecycleLimits
from src.core.data_retention import DataCleanupReport, DataRetentionPolicy
from src.core.experience import LogicalClock
from src.core.inbox_origin import inbox_origin_metadata
from src.core.managed_lifecycle_state import LifecycleCaptureLimits
from src.core.payload_bytes import PayloadCopyLimits
from src.core.payload_ownership import PayloadOwnershipLimits
from src.core.replay_origin import ReplayOriginLimits, ReplayOriginPorts
from src.core.replay_write_origin import ReplayWriteLimits
from src.core.resource_sharing import SharingLimits
from src.core.serving_ports import ServingConfiguration
from src.core.untrained_inbox_origin import untrained_inbox_origin_metadata
from test_expiry_inbox_qualification import (
    FIRST,
    SECOND,
    FakeWork,
    Graph,
    NumericFakeLearner,
    _queue,
    _train,
    _measure,
    _footprint,
    _auxiliary,
    _outside_copy,
    _growth,
    _model_reference,
    _retained,
    _copy_bytes,
    _payloads,
    _fingerprint,
)


CAPTURE_LIMITS = LifecycleCaptureLimits(64, 128)


def test_should_preserve_original_large_policy_integer_domain_before_optional_observation():
    from src.app.expiry_history_birth import expiry_policy_birth

    original = DataRetentionPolicy(
        2**64,
        2**64,
        PayloadOwnershipLimits(16, 64),
        owned_payload_copies=PayloadCopyLimits(2**64),
        max_retention_seconds=2**64,
    )
    snapshot = expiry_policy_birth(original)
    assert snapshot[3][:2] == (2**64, 2**64)
    assert snapshot[4:] == (2**64, 2**64)


@dataclass
class AutomaticWork(FakeWork):
    cleanup_calls: int = 0
    post_erase_probes: int = 0

    def _reserve_arrays(self, count, size):
        assert self.arrays + count <= 512 and self.array_bytes + size <= 64 * 1024
        self.arrays += count
        self.array_bytes += size


@pytest.fixture(scope="module")
def automatic_work(tmp_path_factory):
    work = AutomaticWork()
    try:
        yield work
    finally:
        (tmp_path_factory.mktemp("automatic-expiry-fake-work") / "work.json").write_text(
            json.dumps(asdict(work), indent=2),
            encoding="utf8",
        )
    assert work.graphs <= 24 and work.updates <= 48
    assert work.learners <= 72 and work.forks <= 48
    assert work.arrays <= 512 and work.array_bytes <= 64 * 1024
    assert work.pending_consumed is None
    assert work.graphs == work.updates == 24
    assert work.learners == 72 and work.forks == 48
    assert work.arrays == 156 and work.array_bytes == 1872
    assert work.consumed_arrays == work.native_copy_arrays == 48
    assert work.consumed_bytes == work.native_copy_bytes == 576


class CleanupPorts:
    """Fault controls live behind the SAME original birth-installed ports."""

    def __init__(self, work):
        self.work = work
        self.graph: Graph | None = None
        self.armed = False
        self.on_last_footprint: Callable[[], None] | None = None
        self.native_error: BaseException | None = None
        self.fired = 0

    def footprint(self, learner):
        result = _footprint(learner)
        if (
            self.armed
            and self.graph is not None
            and learner is self.graph.runtime._candidate
            and not learner.model.rows
        ):
            self.work.post_erase_probes += 1
            if self.on_last_footprint is not None:
                action, self.on_last_footprint = self.on_last_footprint, None
                self.fired += 1
                action()
        return result

    def erase(self, learner):
        self.work.cleanup_calls += 1
        if self.armed and self.native_error is not None:
            raise self.native_error
        result = _footprint(learner)
        learner.model.rows.clear()
        return result


def _fresh_automatic(work, *, history=True, records=64, actor=False):
    # Same accepted numeric factory/ports/birth limits, with genuine supported
    # cleanup ports fixed BEFORE lifecycle construction. No later port swapping.
    assert work.graphs + 1 <= 24
    work.graphs += 1
    clock = LogicalClock()
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=2), lambda: 0.0)
    learner = NumericFakeLearner(work)
    original_actor = (
        None
        if not actor
        else PromotableActor(
            learner,
            version="actor-0",
            configuration=ServingConfiguration(2, 2),
            feature_digest=lambda features: "unused",
            metadata={},
        )
    )
    runtime: Any = ActorShadowRuntime(
        learner,
        actor_version="actor-0",
        candidate_version="candidate-0",
        clock=clock,
        budget=budget,
        max_experiences=4,
        actor=original_actor,
    )
    gate = ServingPriorityGate(SharingLimits(1, 2, 1), resource_available=lambda: True)
    owner = ManagedExperienceOwner(
        ResourceSharedRuntime(runtime, gate), limits=LifecycleLimits(4, 4)
    )
    ports = CleanupPorts(work)
    life: Any = ManagedDataLifecycle(
        owner,
        policy=DataRetentionPolicy(
            4096, 10, PayloadOwnershipLimits(16, 64), owned_payload_copies=PayloadCopyLimits(4096)
        ),
        measure_payload_bytes=_measure,
        native_footprint=ports.footprint,
        native_erase=ports.erase,
        measure_auxiliary_bytes=_auxiliary,
        measure_checkpoint_bytes=_outside_copy,
        native_growth_bytes=_growth,
        prepare_model_bytes=_outside_copy,
        prediction_cache_bytes=_outside_copy,
    )
    ledger: Any = ManagedReplayOrigins(
        owner,
        ReplayOriginPorts(_model_reference, _retained, _copy_bytes, _payloads, _fingerprint),
        ReplayOriginLimits(16, records, 8, 128 * 1024, 120, 4096),
        ReplayWriteLimits(4, 2, 64),
        retain_inbox_origins=history,
    )
    graph = Graph(work, owner, life, runtime, ledger, clock, gate, ledger._history())
    ports.graph = graph
    _queue(graph)
    receipt = _train(graph)
    assert receipt is runtime._inbox._applied[FIRST] and receipt.update_number == 1
    assert life._admitted_bytes == 24 and life._copy_budget._charged == 48
    assert runtime._budget.updates_completed == runtime._candidate.calls == 1
    assert life._expiry_history_on is history
    if history:
        assert life._expiry_history_birth is ledger._inbox_history_birth
        assert life._expiry_authority is ledger._expiry_authority_birth
    else:
        assert life._expiry_history_birth is None and ledger._expiry_authority_birth is None
    ports.armed = True
    return graph, ports


def _queue_all_forms(graph):
    _queue(graph, "s2", label=False)
    _queue(graph, "s3", source=False)
    _queue(graph, "s4")
    assert len(graph.owner._catalog) == graph.runtime._inbox._capacity == 4
    assert set(graph.runtime._inbox._applied) == {FIRST}


def _authority(graph):
    ledger, life, runtime = graph.ledger, graph.life, graph.runtime
    return dict(
        admission=ledger._admission,
        limits=ledger._admission.limits,
        raw=life._copy_budget,
        raw_limits=life._copy_budget._limits,
        policy=life._policy,
        budget=runtime._budget,
        work=runtime._budget.updates_completed,
        ingress=life._admitted_bytes,
        charge=life._copy_budget._charged,
        slots=ledger._copy_slots,
        minimum_slots=ledger._minimum_copy_slots,
        enrollments=life._registry._total,
        progress=ledger._admission._progress,
        maps=None
        if graph.history is None
        else (
            graph.history._storage,
            graph.history._erased_storage,
            graph.history._untrained_storage,
        ),
    )


def _assert_authority(graph, before):
    current = _authority(graph)
    for name in ("admission", "limits", "raw", "raw_limits", "policy", "budget"):
        assert current[name] is before[name]
    for name in ("work", "ingress", "charge", "slots", "minimum_slots", "enrollments"):
        assert current[name] == before[name]
    for name in ("records_created", "invocations_started", "metadata_bytes_charged"):
        assert getattr(current["progress"], name) >= getattr(before["progress"], name)


def _weak_payloads(graph):
    return tuple(
        ref(value)
        for mapping, field in (
            (graph.runtime._inbox._experiences, "features"),
            (graph.runtime._inbox._labels, "targets"),
        )
        for record in mapping.values()
        for value in (record, getattr(record, field))
    ) + tuple(
        ref(value)
        for row in graph.runtime._candidate.model.rows
        for value in (row, row.features, row.targets)
    )


def _assert_successful_raw_cleanup(graph, report, before, *, expired_at=122):
    assert observation_module._CURRENT.get() is None
    assert type(report) is DataCleanupReport and report.reason == "expired"
    keys = tuple(sorted(graph.owner._catalog))
    assert report.requested_keys == report.revoked_keys == keys
    assert report.model_snapshots_erased == 1 and report.inboxes_cleared == 1
    assert report.checkpoints_invalidated == report.promotions_invalidated == 0
    assert not graph.runtime._inbox._experiences and not graph.runtime._inbox._labels
    assert not graph.runtime._candidate.model.rows
    assert set(graph.runtime._inbox._applied) == {FIRST}
    assert set(graph.runtime._inbox._erased) == set(keys) == graph.owner._revoked_keys
    assert all(
        value.reason == "expired" and value.erased_at == expired_at
        for value in graph.runtime._inbox._erased.values()
    )
    assert not graph.life._failed and not graph.life._retention_fault and not graph.runtime._stopped
    assert not graph.ledger._poisoned
    _assert_authority(graph, before)


def _assert_unpublished(graph, before):
    history = graph.history
    assert history._storage is before["maps"][0]
    assert history._erased_storage is before["maps"][1]
    assert history._untrained_storage is before["maps"][2]
    assert {record.data.key for record in history._records.values()} == {FIRST}
    assert not history._erased_records and not history._untrained_records


def _observer_refusal(graph, before, *, restore=None):
    with pytest.raises(ExpiryObservationError) as failure:
        graph.life.expire()
    error = failure.value
    assert type(error) is ExpiryObservationError
    assert type(error.code) is str and 0 < len(error.code) <= 64
    assert error.__cause__ is None and error.__context__ is None
    if restore is not None:
        restore()
    _assert_successful_raw_cleanup(graph, error.report, before)
    _assert_unpublished(graph, before)
    return error


def test_should_publish_trained_and_all_three_untrained_forms_only_after_whole_original_expiry(
    automatic_work,
):
    graph, ports = _fresh_automatic(automatic_work)
    _queue_all_forms(graph)
    before = _authority(graph)
    receipt = graph.runtime._inbox._applied[FIRST]
    data = next(iter(graph.history._records.values())).data
    weak = _weak_payloads(graph)

    def still_unpublished():
        _assert_unpublished(graph, before)
        assert graph.runtime._inbox._experiences and graph.runtime._inbox._labels

    ports.on_last_footprint = still_unpublished
    graph.clock.advance_to(122)
    with pytest.raises(ValueError, match="expired"):
        graph.owner._require_live(FIRST)
    report = graph.life.expire()
    _assert_successful_raw_cleanup(graph, report, before)
    assert ports.fired == 1 and all(reference() is None for reference in weak)
    assert not graph.history._records and not graph.history._sealed
    erased = graph.history._erased_records[FIRST]
    assert erased.data is data and erased.receipt() is receipt
    assert erased.tombstone() is graph.runtime._inbox._erased[FIRST]
    assert graph.history._erased_sealed[FIRST] is erased
    assert set(graph.history._untrained_records) == {SECOND, ("e1", "s3"), ("e1", "s4")}
    metadata = len(inbox_origin_metadata(data, 128 * 1024)) + 1024
    for key, record in graph.history._untrained_records.items():
        assert record is graph.history._untrained_sealed[key]
        assert record.tombstone() is graph.runtime._inbox._erased[key]
        assert record.data.key == key and record.data.completed_updates == 1
        assert (
            not hasattr(record, "source")
            and not hasattr(record, "label")
            and not hasattr(record, "receipt")
        )
        metadata += len(untrained_inbox_origin_metadata(record.data, 128 * 1024)) + 1024
    progress = graph.ledger._admission._progress
    assert progress.records_created == before["progress"].records_created + 4
    assert progress.metadata_bytes_charged == before["progress"].metadata_bytes_charged + metadata
    assert progress.invocations_started == before["progress"].invocations_started
    assert (
        ManagedInboxOrigins.verify(graph.history, graph.ledger, graph.owner, graph.runtime, 122)
        == ()
    )


def test_should_return_original_zero_report_when_nothing_is_due_even_with_missing_observation_bridge(
    automatic_work,
):
    graph, ports = _fresh_automatic(automatic_work)
    graph.life._expiry_history_birth = None
    before = _authority(graph)
    cleanup_calls = automatic_work.cleanup_calls
    report = graph.life.expire()
    assert type(report) is DataCleanupReport and report == DataCleanupReport(
        "expired", (), (), 0, 0, 0, 0
    )
    assert graph.runtime._inbox._experiences and graph.runtime._candidate.model.rows
    assert automatic_work.cleanup_calls == cleanup_calls
    assert _authority(graph)["progress"] is before["progress"]
    _assert_unpublished(graph, before)


def test_should_preserve_original_default_off_raw_expiry_without_history_grant(automatic_work):
    graph, ports = _fresh_automatic(automatic_work, history=False)
    before = _authority(graph)
    graph.clock.advance_to(122)
    report = graph.life.expire()
    _assert_successful_raw_cleanup(graph, report, before)
    assert graph.history is None and graph.ledger._inbox_origins is None
    assert graph.life._expiry_history_birth is None and graph.life._expiry_history_on is False


def test_should_remove_raw_for_missing_foreign_or_busy_original_history_without_poisoning(
    automatic_work,
):
    for mode in ("missing", "foreign", "busy"):
        graph, ports = _fresh_automatic(automatic_work)
        before = _authority(graph)
        graph.clock.advance_to(122)
        original = graph.life._expiry_history_birth
        if mode == "missing":
            graph.life._expiry_history_birth = None
        elif mode == "foreign":
            graph.life._expiry_history_birth = ref(graph.life)
        else:
            assert graph.ledger._gate.acquire(blocking=False)
        try:
            _observer_refusal(graph, before)
        finally:
            if mode == "busy":
                graph.ledger._gate.release()
            else:
                graph.life._expiry_history_birth = original
        assert graph.ledger._admission._progress is before["progress"]
        with pytest.raises(ValueError):
            ManagedInboxOrigins.verify(graph.history, graph.ledger, graph.owner, graph.runtime, 122)


def test_should_refuse_equal_valued_replacement_of_original_expiry_birth_tuple(automatic_work):
    graph, ports = _fresh_automatic(automatic_work)
    original = graph.life._expiry_authority
    replacement = tuple(list(original))
    assert replacement == original and replacement is not original
    graph.life._expiry_authority = replacement
    before = _authority(graph)
    graph.clock.advance_to(122)
    _observer_refusal(graph, before)


def test_should_keep_original_partial_record_charges_after_automatic_mixed_reservation_exhaustion(
    automatic_work,
):
    graph, ports = _fresh_automatic(automatic_work, records=3)
    _queue_all_forms(graph)
    before = _authority(graph)
    graph.clock.advance_to(122)
    _observer_refusal(graph, before)
    progress = graph.ledger._admission._progress
    assert progress.records_created == 3 == before["progress"].records_created + 1
    assert progress.metadata_bytes_charged > before["progress"].metadata_bytes_charged
    assert progress.invocations_started == before["progress"].invocations_started


def test_should_refuse_late_original_source_replacement_from_last_native_footprint(automatic_work):
    graph, ports = _fresh_automatic(automatic_work)
    original = graph.runtime._inbox._experiences[FIRST]

    def replace_source():
        graph.runtime._inbox._experiences[FIRST] = replace(original)

    ports.on_last_footprint = replace_source
    before = _authority(graph)
    graph.clock.advance_to(122)
    _observer_refusal(graph, before)
    assert ports.fired == 1


def test_should_refuse_late_original_receipt_replacement_from_last_native_footprint(automatic_work):
    graph, ports = _fresh_automatic(automatic_work)
    original = graph.runtime._inbox._applied[FIRST]
    ports.on_last_footprint = lambda: graph.runtime._inbox._applied.__setitem__(
        FIRST, replace(original)
    )
    before = _authority(graph)
    graph.clock.advance_to(122)
    _observer_refusal(graph, before)
    assert ports.fired == 1 and graph.runtime._inbox._applied[FIRST] is not original


def test_should_preserve_genuine_native_error_priority_over_missing_observation_bridge(
    automatic_work,
):
    graph, ports = _fresh_automatic(automatic_work)
    before = _authority(graph)
    original_error = RuntimeError("declared original fake native erase fault")
    ports.native_error = original_error
    graph.life._expiry_history_birth = None
    graph.clock.advance_to(122)
    with pytest.raises(RuntimeError) as failure:
        graph.life.expire()
    assert failure.value is original_error and not isinstance(failure.value, ExpiryObservationError)
    assert graph.life._failed and graph.runtime._stopped
    _assert_unpublished(graph, before)
    _assert_authority(graph, before)


def test_should_discard_staged_lineage_when_original_report_construction_fails(
    automatic_work, monkeypatch
):
    graph, ports = _fresh_automatic(automatic_work)
    before = _authority(graph)
    original_error = RuntimeError("declared original report construction fault")

    def failed_report(*arguments, **keywords):
        raise original_error

    graph.clock.advance_to(122)
    with monkeypatch.context() as patch:
        patch.setattr(lifecycle_module, "DataCleanupReport", failed_report)
        with pytest.raises(RuntimeError) as failure:
            graph.life.expire()
    assert failure.value is original_error
    assert not graph.runtime._inbox._experiences and not graph.runtime._inbox._labels
    assert not graph.runtime._candidate.model.rows
    _assert_unpublished(graph, before)
    _assert_authority(graph, before)


def test_should_capture_original_history_on_without_ledger_reentry_or_wire_schema_expansion(
    automatic_work,
):
    graph, ports = _fresh_automatic(automatic_work)
    before = _authority(graph)
    # Public capture already holds original lifecycle/owner/runtime leases.
    # A held ledger gate proves capture does not acquire a new history lease.
    assert graph.ledger._gate.acquire(blocking=False)
    try:
        lifecycle = capture_managed_lifecycle(graph.owner, limits=CAPTURE_LIMITS)
        paired = capture_managed_records(graph.owner, limits=CAPTURE_LIMITS)
    finally:
        graph.ledger._gate.release()
    assert lifecycle.metadata.format_version == paired.metadata.format_version == 1
    assert paired.metadata.lifecycle.format_version == 1
    assert len(lifecycle.authority) == 48 and len(paired.authority) == 59
    assert not any("expiry" in item.path for item in lifecycle.authority + paired.authority)
    assert any(
        item.path == "root.lifecycle" and item.value is graph.life for item in lifecycle.authority
    )
    _assert_authority(graph, before)
    assert graph.ledger._admission._progress is before["progress"]


def test_should_refuse_missing_corrupt_or_cloned_birth_fields_at_original_capture(automatic_work):
    graph, ports = _fresh_automatic(automatic_work, actor=True)
    for name in ("_expiry_history_birth", "_expiry_history_on", "_expiry_authority"):
        original = getattr(graph.life, name)
        delattr(graph.life, name)
        try:
            with pytest.raises(ValueError):
                capture_managed_lifecycle(graph.owner, limits=CAPTURE_LIMITS)
        finally:
            setattr(graph.life, name, original)
    for name, replacement in (
        ("_expiry_history_birth", ref(graph.life)),
        ("_expiry_history_on", 1),
        ("_expiry_authority", tuple(list(graph.life._expiry_authority))),
    ):
        original = getattr(graph.life, name)
        setattr(graph.life, name, replacement)
        try:
            with pytest.raises(ValueError):
                capture_managed_records(graph.owner, limits=CAPTURE_LIMITS)
        finally:
            setattr(graph.life, name, original)
    before = _authority(graph)
    original_slot = graph.runtime.actor._slot
    original_generation = original_slot.generation

    def fail_original_promotion_generation():
        object.__setattr__(original_slot, "generation", "invalid-generation")

    ports.on_last_footprint = fail_original_promotion_generation
    graph.clock.advance_to(122)
    try:
        with pytest.raises(TypeError):
            graph.life.expire()
        assert ports.fired == 1 and graph.runtime.actor._slot is original_slot
        assert graph.life._failed and graph.runtime._stopped
        assert not graph.runtime._inbox._experiences and not graph.runtime._inbox._labels
        assert not graph.runtime._candidate.model.rows
        _assert_unpublished(graph, before)
        _assert_authority(graph, before)
    finally:
        object.__setattr__(original_slot, "generation", original_generation)


def test_should_refuse_original_time_work_limits_policy_birth_or_raw_charge_changes_from_last_footprint(
    automatic_work,
):
    for mode in ("time", "work", "limits", "policy", "declaration-birth", "raw-charge"):
        graph, ports = _fresh_automatic(automatic_work)
        graph.clock.advance_to(122)
        before = _authority(graph)
        if mode == "time":
            holder, name, replacement = graph.clock, "_time", 123
        elif mode == "work":
            holder, name, replacement = graph.runtime._budget, "updates_completed", 2
        elif mode == "limits":
            holder, name, replacement = (
                graph.ledger._admission,
                "limits",
                replace(graph.ledger._admission.limits),
            )
        elif mode == "policy":
            holder, name, replacement = graph.life._policy, "max_retention_ticks", 11
        elif mode == "raw-charge":
            holder, name, replacement = (
                graph.life._copy_budget,
                "_charged",
                graph.life._copy_budget._charged - 1,
            )
        else:
            holder, name, replacement = None, None, 1
        if mode == "declaration-birth":
            original = graph.owner._declaration_ticks[FIRST]

            def mutate():
                graph.owner._declaration_ticks[FIRST] = replacement

            def restore():
                graph.owner._declaration_ticks[FIRST] = original
        else:
            assert holder is not None and name is not None
            original = getattr(holder, name)

            def mutate():
                object.__setattr__(holder, name, replacement)

            def restore():
                object.__setattr__(holder, name, original)

        ports.on_last_footprint = mutate
        _observer_refusal(graph, before, restore=restore)
        assert ports.fired == 1 and graph.ledger._admission is before["admission"]


def test_should_refuse_late_native_raw_reinsertion_even_when_last_footprint_returns_zero(
    automatic_work,
):
    graph, ports = _fresh_automatic(automatic_work)
    original_row = graph.runtime._candidate.model.rows[0]
    before = _authority(graph)
    ports.on_last_footprint = lambda: graph.runtime._candidate.model.rows.append(original_row)
    graph.clock.advance_to(122)
    with pytest.raises(ExpiryObservationError) as failure:
        graph.life.expire()
    assert (
        type(failure.value.report) is DataCleanupReport
        and failure.value.report.model_snapshots_erased == 1
    )
    assert ports.fired == 1 and graph.runtime._candidate.model.rows == [original_row]
    assert not graph.runtime._inbox._experiences and not graph.runtime._inbox._labels
    assert not graph.life._failed and not graph.runtime._stopped
    _assert_unpublished(graph, before)
    _assert_authority(graph, before)


def test_should_refuse_raw_inbox_reinsertion_after_pop_before_final_history_publication(
    automatic_work, monkeypatch
):
    graph, ports = _fresh_automatic(automatic_work)
    original_source = graph.runtime._inbox._experiences[FIRST]
    before = _authority(graph)
    reports = []
    original_report = lifecycle_module.DataCleanupReport

    def construct_and_reinsert(*arguments, **keywords):
        report = original_report(*arguments, **keywords)
        assert not graph.runtime._inbox._experiences and not graph.runtime._inbox._labels
        _assert_unpublished(graph, before)
        reports.append(report)
        graph.runtime._inbox._experiences[FIRST] = original_source
        return report

    graph.clock.advance_to(122)
    with monkeypatch.context() as patch:
        # Test-only report boundary: call original constructor once unchanged,
        # then insert the actual old source before deferred lineage publication.
        patch.setattr(lifecycle_module, "DataCleanupReport", construct_and_reinsert)
        with pytest.raises(ExpiryObservationError) as failure:
            graph.life.expire()
    assert len(reports) == 1 and failure.value.report is reports[0]
    assert graph.runtime._inbox._experiences[FIRST] is original_source
    assert not graph.runtime._inbox._labels and not graph.runtime._candidate.model.rows
    assert not graph.life._failed and not graph.runtime._stopped
    _assert_unpublished(graph, before)
    _assert_authority(graph, before)


def test_should_pay_every_mixed_witness_before_first_persistent_allocation_and_publish_after_report(
    automatic_work, monkeypatch
):
    import src.app.expiry_inbox_transition as transition
    import src.app.expiry_untrained_qualification as arrivals

    graph, ports = _fresh_automatic(automatic_work)
    _queue_all_forms(graph)
    before = _authority(graph)
    calls = []
    original_trained = transition.prepare_erased_inbox_origin
    original_untrained = arrivals.prepare_untrained_inbox_origin

    def check_paid():
        assert (
            graph.ledger._admission._progress.records_created
            == before["progress"].records_created + 4
        )
        assert (
            graph.ledger._admission is before["admission"]
            and graph.ledger._admission.limits is before["limits"]
        )
        _assert_unpublished(graph, before)
        assert graph.runtime._inbox._experiences and graph.runtime._inbox._labels

    def trained(*arguments):
        check_paid()
        calls.append("trained")
        return original_trained(*arguments)

    def untrained(*arguments):
        check_paid()
        calls.append("untrained")
        return original_untrained(*arguments)

    reports = []
    original_report = lifecycle_module.DataCleanupReport

    def report(*arguments, **keywords):
        _assert_unpublished(graph, before)
        assert not graph.runtime._inbox._experiences and not graph.runtime._inbox._labels
        result = original_report(*arguments, **keywords)
        reports.append(result)
        return result

    graph.clock.advance_to(122)
    with monkeypatch.context() as patch:
        patch.setattr(transition, "prepare_erased_inbox_origin", trained)
        patch.setattr(arrivals, "prepare_untrained_inbox_origin", untrained)
        patch.setattr(lifecycle_module, "DataCleanupReport", report)
        result = graph.life.expire()
    assert calls == ["untrained", "untrained", "untrained", "trained"]
    assert len(reports) == 1 and result is reports[0]
    _assert_successful_raw_cleanup(graph, result, before)
    assert not graph.history._records and len(graph.history._erased_records) == 1
    assert len(graph.history._untrained_records) == 3


def test_should_refuse_copied_context_scope_access_and_original_cleanup_reentry_without_changing_scope(
    automatic_work,
):
    from contextvars import copy_context
    import src.app.expiry_inbox_observation as observation

    graph, ports = _fresh_automatic(automatic_work)
    before = _authority(graph)
    scopes = []

    def foreign_context_and_reentry():
        scope = observation._CURRENT.get()
        assert scope is not None
        scopes.append(scope)
        names = (
            "life",
            "ledger",
            "owner",
            "runtime",
            "history",
            "gate",
            "staged",
            "fault",
            "closed",
            "used",
            "committed",
            "active",
            "context",
            "access",
            "close_access",
        )
        saved = tuple(getattr(scope, name) for name in names)
        with pytest.raises(ValueError):
            copy_context().run(scope._require_access)
        assert observation._CURRENT.get() is scope
        current = tuple(getattr(scope, name) for name in names)
        assert all(
            current_value is saved_value for current_value, saved_value in zip(current, saved)
        )
        with pytest.raises(ValueError):
            copy_context().run(scope.close, scope.gate, scope.context, scope.close_access)
        assert observation._CURRENT.get() is scope and scope.gate.locked()
        current = tuple(getattr(scope, name) for name in names)
        assert all(
            current_value is saved_value for current_value, saved_value in zip(current, saved)
        )
        cleanup_calls = automatic_work.cleanup_calls
        with pytest.raises(ValueError, match="busy|reentran|nested"):
            graph.life.expire()
        assert automatic_work.cleanup_calls == cleanup_calls
        assert observation._CURRENT.get() is scope
        current = tuple(getattr(scope, name) for name in names)
        assert all(
            current_value is saved_value for current_value, saved_value in zip(current, saved)
        )
        _assert_unpublished(graph, before)

    ports.on_last_footprint = foreign_context_and_reentry
    graph.clock.advance_to(122)
    report = graph.life.expire()
    _assert_successful_raw_cleanup(graph, report, before)
    assert ports.fired == len(scopes) == 1 and observation._CURRENT.get() is None
    assert scopes[0].closed and scopes[0].used and scopes[0].committed
    assert all(
        getattr(scopes[0], name) is None
        for name in (
            "ledger",
            "owner",
            "runtime",
            "history",
            "gate",
            "before",
            "proof",
            "staged",
            "context",
            "access",
            "close_access",
        )
    )
    # No actual contender/thread is needed to prove the copied-context boundary.
    # Root separately reviews the original thread and lexical gate checks.
